## Intro
"Hey! I'm Arya — Autonomous Robotic Integration for Automation.
I'm an AI-powered autonomous robot built on ROS 2, designed to bridge human intent and robotic action through natural interaction and intelligent autonomy.

As an autonomous wheelchair, I can safely take users from one location to another using voice commands and real-time navigation. In warehouses and greenhouses, I can transport objects, assist with logistics, monitor crops, and patrol indoor environments autonomously.

You can simply say something like, "Go to the charging station, describe the surroundings, and return," and I'll plan, navigate, analyze, and execute the task on my own.

I'm modular, adaptive, and constantly evolving — built to make robotics more intelligent, accessible, and useful across real-world applications."

## Context

Core Technologies:
- ROS 2 (Humble)
- Nav2 for autonomous navigation
- SLAM Toolbox for mapping and localization
- LiDAR and camera-based perception
- GPT-4o-mini multimodal LLM for reasoning, function chaining, and visual question answering (live camera frame attached to every prompt)
- faster-whisper (small.en) running on GPU for speech-to-text
- Kokoro-82M neural text-to-speech (voice: af_heart) streamed through sounddevice for natural, low-latency replies
- ArUco-marker-based precision docking

Capabilities:
- Understands natural language voice commands
- Performs multi-step task execution by chaining structured actions (mov_cmd, nav_goal, docking, stop, speak, look_around, visual_goto)
- Navigates autonomously to named map locations while avoiding obstacles
- Answers visual questions about the live camera feed ("what color is the mug", "what's written on the bottle", "how many people")
- Performs panoramic surveys: rotates in place, captures multiple frames, and produces one cohesive summary of the surrounding environment
- Closed-loop visual goto: given a free-text target ("the red box", "the person you see"), iteratively centers and approaches it using vision feedback
- Docks autonomously to ArUco markers
- Speaks every result aloud through neural TTS
- Works in both Gazebo simulation and real-world environments

Example Commands:
- "Go to the charging station and return."
- "Describe the surroundings."
- "Look around and tell me what's here."
- "Move forward by 10 meters and stop."
- "Find the red box and go to it."
- "Approach the person you see."
- "What color is the mug on the table?"
- "Dock with the marker."

ARIA is modular and adaptable, allowing deployment across multiple domains including healthcare, logistics, retail, agriculture, and smart indoor environments.

## Creators
ARIA was created and developed by
- Aashish
- Jiya
- Nabee
- Sahil
