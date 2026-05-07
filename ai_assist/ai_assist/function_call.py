import rclpy, time, os, cv2, yaml,base64, pygame, time, math,  sys, queue, threading, wave, json, numpy as np, sounddevice as sd
from faster_whisper import WhisperModel
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Twist , Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge ,  CvBridgeError
from nav_msgs.msg import Odometry
from dotenv import load_dotenv
from openai import OpenAI
from nav2_msgs.action import NavigateToPose
from kokoro import KPipeline
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory

if not hasattr(np, 'float'):
    np.float = float  # patch deprecated alias


from tf_transformations import quaternion_from_euler, euler_from_quaternion 
from std_srvs.srv import Trigger 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from function_tools import first_tools, second_tools
from dock import FollowAruco

load_dotenv()


class Prompt(Node):
    def __init__(self):
        super().__init__('ai_assist_engine')
        pygame.init()

        self.tts_pipeline = KPipeline(lang_code="a")
        self.tts_voice = "af_heart"
        self.tts_sample_rate = 24000
        self.tts_stream = sd.OutputStream(
            samplerate=self.tts_sample_rate, channels=1, dtype="float32"
        )
        self.tts_stream.start()

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        )
        self.bridge = CvBridge()
        self.cv_image = None

        self.sub_aruco = self.create_subscription(
            Point, '/detected_marker', self.listener_callback, 10)

        self.client = OpenAI(api_key=os.getenv('API_KEY'))

        self.timer = None

        # Whisper & Audio setup
        import torch
        whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if whisper_device == "cuda" else "int8"
        self.get_logger().info(f"Loading faster-whisper small.en on {whisper_device} ({compute_type})...")
        self.model = WhisperModel("small.en", device=whisper_device, compute_type=compute_type)
        self.SAMPLE_RATE = 44100
        self.FILENAME = "recorded_audio.wav"
        self.audio_queue = queue.Queue()
        self.recording = False
        self.transcribed_text = "Press SPACE to record, release to transcribe."
        self.frames = []

        # Pygame GUI
        self.WIDTH, self.HEIGHT = 600, 400
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Speech Command to Robot Actions")
        self.BG_COLOR = (14, 17, 23)
        self.TEXT_COLOR = (250, 250, 250)
        self.font = pygame.font.Font(None, 30)
        self.follow_node = FollowAruco()

        self.declare_parameter("stop_distance", 5.0)  # Stop at 20 cm
        self.stop_distance = self.get_parameter('stop_distance').value

        self.target_x = 0.0
        self.target_z = 1000.0  # Start with a large distance
        self.last_received_time = time.time() - 10000

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose') 


        file_path = os.path.join(
            get_package_share_directory('ai_assist'), 'config', 'config.yaml'
        )

        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)

        self.pre_dock_position = {}
        for item in config_data['pre_dock_position']:
            for key, value in item.items():
                self.pre_dock_position[key] = {'x': value[0], 'y': value[1], 'yaw': value[2]}

        self.aria_context = ""
        context_path = os.path.join(
            get_package_share_directory('ai_assist'), 'aria_context.md'
        )
        try:
            with open(context_path, 'r') as f:
                self.aria_context = f.read()
            self.get_logger().info(f"Loaded ARIA context from {context_path}")
        except FileNotFoundError:
            self.get_logger().warn(
                f"aria_context.md not found at {context_path}; running without identity context."
            )

        self.pose_subscription = self.create_subscription(
            Odometry,  
            'odom',  
            self.pose_callback,
            10)

        self.current_pose = None
        self.get_logger().info('Waiting for nav action server (5s)...')
        if self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().info('Nav action server ready.')
        else:
            self.get_logger().warn('Nav action server not available — nav_goal will fail until Nav2 is running.')


    def image_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")

    def get_current_frame_b64(self, size=(320, 180), quality=80):
        if self.cv_image is None:
            return None
        resized = cv2.resize(self.cv_image, size)
        ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def draw_text(self, text):
        self.screen.fill(self.BG_COLOR)
        words = text.split(" ")
        lines, line = [], ""
        for word in words:
            test_line = line + word + " "
            if self.font.size(test_line)[0] < self.WIDTH - 40:
                line = test_line
            else:
                lines.append(line)
                line = word + " "
        lines.append(line)
        y = self.HEIGHT // 2 - (len(lines) * 15)
        for line in lines:
            text_surface = self.font.render(line, True, self.TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(self.WIDTH // 2, y))
            self.screen.blit(text_surface, text_rect)
            y += 30
        pygame.display.flip()

    def audio_callback(self, indata, frames_count, time, status):
        if status:
            print("Audio Status:", status)
        self.audio_queue.put(indata.copy())

    def start_recording(self):
        self.frames = []
        self.audio_queue.queue.clear()
        self.draw_text("Listening...")
        self.recording = True

        def threaded_record():
            with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, dtype=np.int16, callback=self.audio_callback):
                while self.recording:
                    while not self.audio_queue.empty():
                        self.frames.append(self.audio_queue.get())

        threading.Thread(target=threaded_record, daemon=True).start()

    def stop_recording(self):
        self.recording = False

    def _call_vision(self, system, prompt, frames=None, json_mode=False, model="gpt-4o-mini", temperature=0):
        user_content = [{"type": "text", "text": prompt}]
        for fb64 in (frames or []):
            if not fb64:
                continue
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{fb64}", "detail": "low"},
            })
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()

    def get_gpt_response(self, prompt):
        try:
            img_b64 = self.get_current_frame_b64()
            if img_b64:
                self.get_logger().info("Including current camera frame as context.")
            else:
                self.get_logger().warn("No camera frame available — sending text only.")

            system_prompt = (
                "You are ARIA, a friendly autonomous mobile robot. You receive a live camera image "
                "showing what you currently see. Convert the user's command into a JSON list of actions. "
                "Available actions:\n"
                "  - mov_cmd: relative movement, fields 'linear_x' (meters, positive=forward) "
                "and 'angular_z' (radians, positive=left). Use for simple motion commands "
                "('turn left', 'back up one meter'). Multiple mov_cmd actions can be chained.\n"
                "  - nav_goal: navigate to a named map location, field 'location'. ONLY use the EXACT keys: "
                f"{list(self.pre_dock_position.keys())}. Never invent locations or pass null.\n"
                "  - docking: dock to an ArUco marker (no fields)\n"
                "  - stop: halt all motion\n"
                "  - speak: reply with text, field 'text'. USE THIS for ANY question or conversational request "
                "(e.g. 'what do you see', 'what color is the mug', 'what's written on the bottle', 'how many people', "
                "'where is the door'). Look at the attached camera image and answer the user's SPECIFIC question "
                "directly in 1-2 short sentences. Do NOT just describe the whole scene unless that's what was asked. "
                "If you can't tell from the image, say so honestly.\n"
                "  - look_around: rotate in place, capturing several frames, then summarize the surrounding environment. "
                "Optional fields 'n_frames' (default 4) and 'total_angle_rad' (default ~6.28 = full circle). "
                "Use this when the user wants a panoramic survey ('look around', 'scan the room', 'check what's behind you').\n"
                "  - visual_goto: drive toward an object/person visible (or expected to be visible) in the camera, "
                "field 'target' (free-text description, e.g. 'the red box', 'the person', 'the door'). "
                "Use this for ANY 'go to / approach / find / move toward X' command where X is a visual target rather "
                "than a named map location. The robot will close a vision loop until it reaches the target.\n"
                f"Predefined locations: {self.pre_dock_position}. "
                'Respond with a JSON object of EXACTLY this shape: '
                '{"actions": [{"action": "<name>", ...fields}]}. '
                'Examples:\n'
                '{"actions":[{"action":"speak","text":"I see a desk and a chair."}]}\n'
                '{"actions":[{"action":"mov_cmd","linear_x":1.0,"angular_z":0.0},'
                '{"action":"mov_cmd","linear_x":0.0,"angular_z":-1.57}]}\n'
                '{"actions":[{"action":"nav_goal","location":"burger_king"}]}\n'
                '{"actions":[{"action":"look_around"}]}\n'
                '{"actions":[{"action":"look_around","n_frames":8}]}\n'
                '{"actions":[{"action":"visual_goto","target":"the person you see"}]}\n'
                'Every action object MUST have an "action" key with the action name as its string value. '
                'Do NOT use the action name as the key.'
            )

            if self.aria_context:
                system_prompt += (
                    "\n\n=== About yourself ===\n"
                    "The text below is your background reference, organized in markdown sections.\n"
                    "- When the user asks you to 'introduce yourself' / 'who are you' / "
                    "'tell me about yourself' / 'what are you', emit a single speak action whose "
                    "'text' field is the `## Intro` section reproduced VERBATIM (word-for-word, "
                    "including the opening 'Hey! I'm A.R.I.A...').\n"
                    "- For any other identity / capability / creator / technology question, "
                    "use the `## Context` and `## Creators` sections as background and paraphrase "
                    "naturally in 1-2 short sentences for TTS.\n"
                    "- Note: the user's microphone is transcribed by speech-to-text and your name "
                    "'ARIA' is often mis-heard as 'arya', 'aria', 'area', 'aaria', or similar phonetic "
                    "variants — treat ALL such forms as referring to YOU, never as separate words "
                    "or unknown entities.\n"
                    f"---\n{self.aria_context}"
                )

            content = self._call_vision(
                system=system_prompt,
                prompt=prompt,
                frames=[img_b64] if img_b64 else [],
                json_mode=True,
            )
            print(f"GPT raw: {content}")
            if not content:
                print("GPT returned empty content.")
                return None
            data = json.loads(content)
            actions = data.get("actions") if isinstance(data, dict) else data
            return actions if isinstance(actions, list) else None
        except Exception as e:
            print(f"GPT Error: {e}")
            return None

    def move(self, distance=0.0, angle=0.0):
        linear_speed = 1.0     # meters per second
        angular_speed = 0.5    # radians per second
        msg = Twist()

        # Calculate duration
        linear_duration = abs(distance) / linear_speed if distance != 0 else 0
        angular_duration = abs(angle) / angular_speed if angle != 0 else 0

        # Move linearly
        if distance != 0:
            msg.linear.x = linear_speed if distance > 0 else -linear_speed
            msg.angular.z = 0.0
            start_time = time.time()
            while time.time() - start_time < linear_duration:
                self.publisher_.publish(msg)
                time.sleep(0.1)

        # Rotate
        if angle != 0:
            msg.linear.x = 0.0
            msg.angular.z = angular_speed if angle > 0 else -angular_speed
            start_time = time.time()
            while time.time() - start_time < angular_duration:
                self.publisher_.publish(msg)
                time.sleep(0.1)

        # Stop the robot
        self.stop_robot()


    def stop_robot(self):
        time.sleep(1)
        self.publisher_.publish(Twist())  # Stop after a short move

    def look_around(self, n_frames=4, total_angle_rad=2 * math.pi):
        try:
            n_frames = max(1, int(n_frames))
        except (TypeError, ValueError):
            n_frames = 4
        try:
            total_angle_rad = float(total_angle_rad)
        except (TypeError, ValueError):
            total_angle_rad = 2 * math.pi
        step = total_angle_rad / n_frames
        self.get_logger().info(
            f"Look-around: {n_frames} frames over {math.degrees(total_angle_rad):.0f}° "
            f"(step {math.degrees(step):.0f}°)"
        )

        frames = []
        for i in range(n_frames):
            time.sleep(0.3)  # let the camera settle after motion
            fb64 = self.get_current_frame_b64()
            if fb64:
                frames.append(fb64)
            if i < n_frames - 1:
                self.move(0.0, step)

        if not frames:
            return "I couldn't capture any camera frames while looking around."

        try:
            summary = self._call_vision(
                system=(
                    "You are ARIA's perception system. Several frames were captured at evenly "
                    "spaced angles while the robot rotated in place. Combine them into a 2-3 "
                    "sentence summary of the surrounding environment, starting with 'I see'. "
                    "Mention notable objects on different sides if relevant. Do not list each "
                    "frame separately — give one cohesive summary."
                ),
                prompt=f"{len(frames)} frames captured at {math.degrees(step):.0f}° intervals.",
                frames=frames,
                json_mode=False,
            )
        except Exception as e:
            self.get_logger().error(f"Look-around summary failed: {e}")
            return "I looked around but couldn't summarize what I saw."
        return summary or "I looked around but didn't see much worth mentioning."

    def visual_goto(self, target, max_iters=8, deadline_s=30.0):
        if not target:
            return "I need a target to go to."
        self.get_logger().info(f"Visual goto: {target!r}")
        deadline = time.time() + deadline_s

        for it in range(max_iters):
            if time.time() > deadline:
                self.get_logger().warn("Visual goto: deadline exceeded")
                break

            fb64 = self.get_current_frame_b64()
            if not fb64:
                return "I don't have a camera image to work with."

            try:
                raw = self._call_vision(
                    system=(
                        "You are vision for a ground robot servoing toward a target shown in "
                        "the camera image. Reply with a JSON object: {"
                        "\"found\": boolean, "
                        "\"x_offset\": number in [-1,1] (negative=left, positive=right) for the target's "
                        "horizontal position relative to the image center, "
                        "\"forward_m\": number (suggested forward step in meters; 0 if at target or unsure), "
                        "\"reached\": boolean (true if the robot is close to and centered on the target), "
                        "\"lost\": boolean (true if the target is not visible in this frame), "
                        "\"note\": short string status}."
                    ),
                    prompt=f"Target to approach: {target}",
                    frames=[fb64],
                    json_mode=True,
                )
                obs = json.loads(raw) if raw else {}
            except Exception as e:
                self.get_logger().error(f"Visual goto vision call failed: {e}")
                return f"Vision failed while approaching {target}."

            self.get_logger().info(f"  iter {it}: {obs}")

            if obs.get("reached"):
                return f"Reached {target}."
            if obs.get("lost") or not obs.get("found", False):
                self.move(0.0, math.pi / 4)  # search rotate ~45° left
                continue

            try:
                x_off = float(obs.get("x_offset", 0.0))
            except (TypeError, ValueError):
                x_off = 0.0
            if abs(x_off) > 0.10:
                self.move(0.0, -x_off * 0.6)  # turn proportional, gentle

            try:
                step_m = float(obs.get("forward_m", 0.0))
            except (TypeError, ValueError):
                step_m = 0.0
            step_m = max(0.0, min(step_m, 0.5))
            if step_m > 0.0:
                self.move(step_m, 0.0)

        return f"Could not reach {target}."

    def speak(self, text, language='en'):
        if not text:
            return
        for _, _, audio in self.tts_pipeline(text, voice=self.tts_voice, speed=1.0):
            self.tts_stream.write(np.asarray(audio, dtype=np.float32))

    def _normalize_action(self, action):
        if not isinstance(action, dict):
            return {}
        if action.get("action"):
            return action
        known = {"mov_cmd", "nav_goal", "docking", "stop", "speak", "look_around", "visual_goto"}
        for key, val in action.items():
            if key in known:
                normalized = {"action": key}
                if isinstance(val, dict):
                    normalized.update(val)
                elif isinstance(val, str):
                    if key == "speak":
                        normalized["text"] = val
                    elif key == "nav_goal":
                        normalized["location"] = val
                    elif key == "visual_goto":
                        normalized["target"] = val
                return normalized
        return action

    def call_function_based_on_command(self, command):
        actions = self.get_gpt_response(command)
        if actions is None:
            return "GPT returned invalid format."
        if not actions:
            return ["I'm not sure what to do."]
        results = []
        for action in actions:
            action = self._normalize_action(action)
            action_type = action.get("action")
            if action_type == "mov_cmd":
                self.move(float(action.get("linear_x", 0)), float(action.get("angular_z", 0)))
                results.append(f"Moved {action.get('linear_x', 0)} meters , rotated {action.get('angular_z', 0)} radians.")
            elif action_type == "nav_goal":
                location = action.get("location")
                if not location or location not in self.pre_dock_position:
                    results.append(f"Unknown location '{location}'. Known: {list(self.pre_dock_position.keys())}")
                else:
                    print("Moving to:", location)
                    self.move_to_goal(location)
                    results.append(f"Moved to {location}.")
            elif action_type == "docking":
                if self.timer is not None:
                    self.timer.cancel()
                self.timer = self.create_timer(0.1, self.timer_callback_aruco)
                results.append("Docking with marker.")
            elif action_type == "speak":
                results.append(action.get("text", ""))
            elif action_type == "stop":
                self.stop_robot()
                results.append("Stopped.")
            elif action_type == "look_around":
                results.append(self.look_around(
                    n_frames=action.get("n_frames", 4),
                    total_angle_rad=action.get("total_angle_rad", 2 * math.pi),
                ))
            elif action_type == "visual_goto":
                results.append(self.visual_goto(action.get("target")))
            else:
                results.append(f"Unknown action: {action_type}")
        return results

    def stop_and_transcribe(self):
        self.draw_text("Processing...")
        if self.frames:
            with wave.open(self.FILENAME, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(np.concatenate(self.frames).astype(np.int16).tobytes())

            try:
                segments, _ = self.model.transcribe(self.FILENAME, beam_size=5, vad_filter=True)
                self.transcribed_text = " ".join(s.text for s in segments).strip()
            except Exception as e:
                self.transcribed_text = "Error in transcription."
                print("Transcription error:", e)
        else:
            self.transcribed_text = "No audio captured."

        self.draw_text(f"You said: {self.transcribed_text}")
        print("Final Transcription:", self.transcribed_text)
        result = self.call_function_based_on_command(self.transcribed_text)
        print("Function Output:")
        if isinstance(result, str):
            print(f"  {result}")
            self.speak(result)
        else:
            for step, action in enumerate(result, start=1):
                print(f"  Step {step}: {action}")
                self.speak(action)

    def process_voice_command(self):
        self.proc()

    def proc(self):
        running = True
        last_displayed_text = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.recording:
                        self.start_recording()

                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE and self.recording:
                        self.stop_recording()
                        self.stop_and_transcribe()

            if self.transcribed_text != last_displayed_text:
                self.draw_text(self.transcribed_text)
                last_displayed_text = self.transcribed_text

            pygame.time.delay(100)

    def timer_callback_aruco(self):
        msg = Twist()

        # If marker detected recently
        if (time.time() - self.last_received_time < 1.0):
            self.get_logger().info(f'Target: {self.target_x}, Distance: {self.target_z:.2f} cm')

            if self.target_z > self.stop_distance:
                msg.linear.x = 0.3  # Move forward
            else:
                self.get_logger().info('Reached target distance. Stopping.')
                msg.linear.x = 0.0  # Stop movement
                if self.timer is not None:
                    self.timer.cancel()
                self.timer = None

            msg.angular.z = -0.7 * self.target_x  # Rotate to align with marker
        else:
            self.get_logger().info('Target lost. Searching...')
            msg.angular.z = 0.5  # Rotate in place

        self.publisher_.publish(msg)

    def listener_callback(self, msg):
        self.target_x = msg.x
        self.target_z = msg.z
        self.last_received_time = time.time()

    def pose_callback(self, msg):
        """Callback to handle the robot's current pose."""
        self.current_pose = msg.pose.pose  

    def get_current_pose(self):
        """Returns the robot's current pose as (x, y, yaw in radians)"""
        if self.current_pose:
            x = self.current_pose.position.x
            y = self.current_pose.position.y
            orientation_q = self.current_pose.orientation
            _, _, yaw = euler_from_quaternion(
                [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
            return x, y, yaw
        return None

    def send_goal(self, x, y, yaw):
        """Sends a navigation goal to the action server."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        q = self.quaternion_from_yaw(yaw)
        goal_msg.pose.pose.orientation.x = q[0]
        goal_msg.pose.pose.orientation.y = q[1]
        goal_msg.pose.pose.orientation.z = q[2]
        goal_msg.pose.pose.orientation.w = q[3]

        yaw_degrees = math.degrees(yaw)
        self.get_logger().info(f'Sending goal to x: {x}, y: {y}, yaw: {yaw_degrees} degrees')
        return self.nav_client.send_goal_async(goal_msg)

    def quaternion_from_yaw(self, yaw):
        """Helper function to create a quaternion from yaw."""
        return quaternion_from_euler(0, 0, yaw)

    def check_goal_status(self, goal_handle_future):
        goal_handle = goal_handle_future.result()

        if goal_handle is None:
            self.get_logger().error("Goal handle future returned None. Possible communication failure.")
            return False

        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected :(')
            return False

        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()

        start_time = time.time()
        timeout_sec = 30.0  # Adjust as needed

        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout_sec:
                self.get_logger().warn('Timed out waiting for result.')
                return False

        result = result_future.result()

        if result.status == 4:
            self.get_logger().info('Goal succeeded!')
            current_pose = self.get_current_pose()
            if current_pose:
                x, y, yaw = current_pose
                yaw_degrees = math.degrees(yaw)
                self.get_logger().info(f'Robot reached position: x: {x}, y: {y}, yaw: {yaw_degrees} degrees')
            else:
                self.get_logger().info('Unable to retrieve current pose')
            return True
        else:
            self.get_logger().info(f'Goal failed with status code: {result.status}')
            return False



    def move_to_goal(self, name):
        if name not in self.pre_dock_position:
            self.get_logger().error(f"Location '{name}' not found in predefined positions.")
            return
        """Move to a specific waypoint and perform actions at the waypoint."""
        coordinates = self.pre_dock_position[name]
        x, y, yaw = coordinates['x'], coordinates['y'], coordinates['yaw']
        self.get_logger().info(f"Navigating to {name} (x: {x}, y: {y}, yaw: {yaw})")

        future = self.send_goal(x, y, yaw)
        print("goal sent")
        # rclpy.spin_until_future_complete(self, future)

        if not self.check_goal_status(future):
            self.get_logger().info(f'Navigation failed at {name}, stopping.')
            return 
        else:
            self.get_logger().info(f'Reached {name} successfully.')

def clean_exit():
    print("\nExiting cleanly...")
    pygame.quit()
    sys.exit()

def main(args=None):
    rclpy.init(args=args)
    prompt_engine = Prompt()

    spin_thread = threading.Thread(target=rclpy.spin, args=(prompt_engine,), daemon=True)
    spin_thread.start()

    try:
        prompt_engine.proc()
    except KeyboardInterrupt:
        print("Shutting down due to KeyboardInterrupt.")
    finally:
        try:
            prompt_engine.tts_stream.stop()
            prompt_engine.tts_stream.close()
        except Exception:
            pass
        prompt_engine.destroy_node()
        rclpy.shutdown()
        # clean_exit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clean_exit()

