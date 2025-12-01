# AI Tools for ROS2

A ROS2 Python package that enables AI-powered voice command control for mobile robots. This package integrates OpenAI's GPT models for natural language understanding, Whisper for speech-to-text conversion, and gTTS for text-to-speech responses.

## Features

- **Voice Command Control**: Control your robot using natural language voice commands
- **AI-Powered Movement**: Uses GPT-4o-mini to interpret commands and translate them into robot movements
- **Image Capture & Description**: Capture images from the robot's camera and get AI-generated descriptions
- **Text-to-Speech Feedback**: Robot responds with audio feedback using Google Text-to-Speech
- **Speech Recognition**: Real-time speech-to-text conversion using OpenAI Whisper
- **Function Calling**: Structured function calls for precise robot control

## Supported Commands

- **Movement Commands**: forward, back, left, right
- **Image Capture**: "capture" - takes an image and describes what the robot sees

## Prerequisites

- ROS2 Humble or later
- Python 3.8+
- A robot with `/cmd_vel` topic for movement control
- A camera publishing to `/camera/image_raw/uncompressed` topic

## Dependencies

### ROS2 Dependencies

- `rclpy`
- `geometry_msgs`
- `sensor_msgs`
- `cv_bridge`

## Installation

1. Clone this repository into your ROS2 workspace:

   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/JiyaPandey/ai_tools.git
   ```

2. Install Python dependencies:

   ```bash
   pip install openai python-dotenv opencv-python pygame gtts pillow openai-whisper sounddevice numpy
   ```

3. Create a `.env` file in the package root with your API key:

   ```bash
   API_KEY=your_openai_api_key_here
   ```

4. Build the package:

   ```bash
   cd ~/ros2_ws
   colcon build --packages-select ai_tools
   source install/setup.bash
   ```

## Usage

### Running the Prompt Engine

The main node for voice/text command control:

```bash
ros2 run ai_tools prompt
```

This starts the prompt engine which:
- Subscribes to camera images on `/camera/image_raw/uncompressed`
- Publishes velocity commands to `/cmd_vel`
- Accepts text input for robot control commands

### Running the Image Prompt Node

For image-only processing:

```bash
ros2 run ai_tools img_prompt
```

## Package Structure

```
ai_tools/
├── ai_tools/
│   ├── __init__.py
│   ├── prompt.py           # Main prompt engine node
│   ├── img_prompt.py       # Image prompt processing node
│   ├── function_call.py    # Enhanced prompt engine with function calling
│   ├── xyz.py              # Standalone voice command recorder with GUI
│   └── test.py             # Speech-to-text testing utilities
├── resource/
│   └── ai_tools
├── test/
│   ├── test_copyright.py
│   ├── test_flake8.py
│   └── test_pep257.py
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

## How It Works

1. **Voice/Text Input**: The system accepts natural language commands either through voice (using Whisper) or text input
2. **AI Processing**: Commands are sent to OpenAI's GPT model which interprets the intent
3. **Command Translation**: The AI translates natural language into robot commands (front, back, left, right, capture)
4. **Robot Control**: Commands are converted to ROS2 Twist messages and published to `/cmd_vel`
5. **Image Description**: When "capture" is commanded, the robot takes an image and uses GPT-4o-mini vision to describe what it sees
6. **Audio Feedback**: Descriptions are converted to speech using gTTS and played back

## Configuration

### Environment Variables

Create a `.env` file with the following:

```env
API_KEY=your_openai_api_key
MY_KEY=your_openai_api_key
```

> **Note**: Both `API_KEY` and `MY_KEY` should be set to the same OpenAI API key. `API_KEY` is used by the ROS2 nodes (`prompt.py`, `function_call.py`), while `MY_KEY` is used by standalone scripts (`xyz.py`, `test.py`).

### ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel` | `geometry_msgs/Twist` | Robot velocity commands (published) |
| `/camera/image_raw/uncompressed` | `sensor_msgs/Image` | Camera input (subscribed) |
| `/image_in` | `sensor_msgs/Image` | Alternative camera input for img_prompt |


## License

See package.xml for license information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
