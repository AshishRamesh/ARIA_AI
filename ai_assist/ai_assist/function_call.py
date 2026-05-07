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
from PIL import Image as PILImage
from kokoro import KPipeline
from io import BytesIO
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
            "/camera/image_raw/uncompressed",
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
        """Receives and converts camera images."""
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  
            self.get_logger().info("Image received.")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")

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

    def get_gpt_response(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are controlling a robot. Convert the given command into a structured JSON list of actions. "
                           "Each action must have 'action' (mov_cmd, nav_goal, docking, capture, or stop). "
                           "If these 4 location (petrol pump,burger king,apartment area and parking) is mentioned, use 'nav_goal'. If just movement is mentioned, use 'mov_cmd'. "
                           f"Available predefined locations: {self.pre_dock_position}"
                           "Return ONLY JSON, with no extra text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                tools=first_tools + second_tools,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.strip("```json").strip("```")
            actions = json.loads(content)
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

    def capture(self):
        image_file = self.capture_image()
        if image_file:
            self.send_image_for_description(image_file)
            return "Captured and described image."
        return "Failed to capture image."


    def capture_image(self, filename='captured_image.jpg'):
        """Captures an image from the ROS2 camera topic and resizes it before saving."""
        if self.cv_image is None:
            self.get_logger().error("No image received yet. Waiting for image...")
            rclpy.spin_once(self, timeout_sec=2.0)
            if self.cv_image is None:
                self.get_logger().error("Still no image received. Cannot capture.")
                return None

        try:
            resized_image = cv2.resize(self.cv_image, (640, 360))
            cv2.imwrite(filename, resized_image)
            self.get_logger().info(f"Image saved as {filename} (Resized to 640x360)")
            return filename
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")
            return None


    def encode_image(self, image_path):
        """Encodes an image to Base64 format for OpenAI API."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.get_logger().error(f"Error encoding image: {e}")
            return None

    def resize_and_compress_image(self,image_path, output_path='compressed_image.jpg', size=(320, 180), quality=80):
        """Resizes and compresses the image to reduce file size."""
        with PILImage.open(image_path) as img:
            img = img.resize(size, PILImage.LANCZOS) 
            img.save(output_path, "JPEG", quality=quality)  
        return output_path

    def send_image_for_description(self, image_file):
        """Sends the captured image to OpenAI's API for description."""
        if image_file:
            compressed_image = self.resize_and_compress_image(image_file)
        
        base64_image = self.encode_image(compressed_image)
        if not base64_image:
            return

        try:
            response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Autonomous Mobile Robot and can describe what you see keep it short. "
                 "Also give response starting with I see and you dont have to mention if the image is blur or dim "},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=500,
            n=1,
            )
            description = response.choices[0].message.content.strip().lower()
            self.get_logger().info(f"Image Description: {description}")
            self.draw_text(f"Image Description: {description}")
            self.speak(description)
        except Exception as e:
            self.get_logger().error(f"Error during OpenAI API request: {e}")
        finally:
            # Clean up the image files
            for file_path in [image_file, compressed_image]:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        self.get_logger().info(f"Deleted temporary image file: {file_path}")
                except Exception as e:
                    self.get_logger().warn(f"Could not delete {file_path}: {e}")

    def speak(self, text, language='en'):
        if not text:
            return
        for _, _, audio in self.tts_pipeline(text, voice=self.tts_voice, speed=1.0):
            self.tts_stream.write(np.asarray(audio, dtype=np.float32))

    def call_function_based_on_command(self, command):
        actions = self.get_gpt_response(command)
        if actions is None:
            return "GPT returned invalid format."
        results = []
        for action in actions:
            action_type = action.get("action")
            if action_type == "mov_cmd":
                self.move(float(action.get("linear_x", 0)), float(action.get("angular_z", 0)))
                results.append(f"Moved {action.get('linear_x', 0)} meters , rotated {action.get('angular_z', 0)} radians.")
            elif action_type == "capture":
                results.append(self.capture())
            elif action_type == "nav_goal":
                location = action.get("location")
                print("Moving to:", location)
                self.move_to_goal(location)
                results.append(f"Moved to {location}.")
            elif action_type == "docking":
                if self.timer is not None:
                    self.timer.cancel()
                self.timer = self.create_timer(0.1, self.timer_callback_aruco)
                results.append("Docking with marker.")
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

