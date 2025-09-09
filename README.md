<div align="center">

<img width="3876" height="719" alt="fbot_vision" src="https://github.com/user-attachments/assets/845c6a57-4888-4036-92bc-6d978ad1d721" />

![UBUNTU](https://img.shields.io/badge/UBUNTU-22.04-orange?style=for-the-badsge&logo=ubuntu)
![python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badsge&logo=python)
![ROS2](https://img.shields.io/badge/ROS2-Humble-blue?style=for-the-badsge&logo=ros)
[![Last Commit](https://img.shields.io/github/last-commit/fbotathome/fbot_vision.svg?style=for-the-badsge)](https://github.com/fbotathome/fbot_vision/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/fbotathome/fbot_vision)](https://github.com/fbotathome/fbot_vision/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/fbotathome/fbot_vision)](https://github.com/fbotathome/fbot_vision/pulls)
[![Contributors](https://img.shields.io/github/contributors/fbotathome/fbot_vision.svg)](https://github.com/fbotathome/fbot_vision/graphs/contributors)

**A ROS 2 vision system for robotics applications featuring object detection, face recognition, person tracking, and vision-language model integration.**

[Overview](#overview) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [fbot_vision message and services](#fbot_vision-message-and-services) • [Contributing](#contributing)

</div>

## Overview

```fbot_vision``` is a ROS 2 package suite designed for robotic vision applications. It provides real-time object detection, face recognition, person tracking with pose estimation, and vision-language model capabilities for interactive robotics systems. It was designed for the RoboCup@Home and the robot BORIS competition but is adaptable to various robotics scenarios.

---

## Architecture

The system consists of three main packages:

```
fbot_vision/
├── 📁 fbot_recognition/          # Core recognition algorithms
|   ├── 📁 base_recognition/      # Abstract base class for all recognition modules
│   ├── 📁 face_recognition/      # Face detection and recognition
│   ├── 📁 moondream_recognition/ # Object recognition using VLM Moondream2
│   ├── 📁 yolo_tracker_recognition/ # People tracking
│   └── 📁 yolov8_recognition/    # Object detection with YOLOv8
├── 📁 fbot_vlm/                  # Vision Language Model integration
└── 📁 fbot_vision_msgs/          # Custom ROS message definitions
```

---

## Installation

### Prerequisites

- ROS2 Humble
- Python 3.10+
- Ubuntu 22.04
- Dependencies listed in `package.xml` and `requirements.txt`

### Setup

1. **Clone the repository into your ROS workspace:**
   ```bash
   cd ~/fbot_ws/src
   git clone https://github.com/fbotathome/fbot_vision.git
   ```

2. **Install dependencies:**
   ```bash
   cd ~/fbot_ws
   sudo rosdep init  # Skip if already initialized
   rosdep update
   rosdep install --from-paths src --ignore-src -r -y
   pip install -r src/fbot_vision/requirements.txt
   ```

3. **Build the workspace:**
   ```bash
   cd ~/fbot_ws
   colcon build --packages-select fbot_recognition fbot_vlm fbot_vision_msgs
   source install/setup.bash
   ```

---

## Usage

### Object Detection
```bash
# Launch YOLOv8 object detection
ros2 launch fbot_recognition yolov8_object_recognition.launch.py use_realsense:=True

# Start/stop detection service
ros2 service call /fbot_vision/fr/object_start std_srvs/srv/Empty
ros2 service call /fbot_vision/fr/object_stop std_srvs/srv/Empty
```

### Person Tracking
```bash
# Launch YOLO tracker with pose estimation
ros2 launch fbot_recognition yolo_tracker_recognition.launch.py use_realsense:=True
 
# Start/stop tracking
ros2 service call /fbot_vision/pt/start std_srvs/srv/Empty
ros2 service call /fbot_vision/pt/stop std_srvs/srv/Empty
```

### Face Recognition
```bash
# Launch face recognition
ros2 launch fbot_recognition face_recognition.launch.py

# Introduce a new person
ros2 service call /fbot_vision/face_recognition/people_introducing \
    fbot_vision_msgs/srv/PeopleIntroducing "{name: 'John Doe'}"

# Forget an existing person from database
ros2 service call /fbot_vision/face_recognition/people_forgetting \
    fbot_vision_msgs/srv/PeopleForgetting "{name: 'John Doe'}"

```


### Moondream Object Recognition
```bash
# Launch Moondream object recognition (local)
ros2 launch fbot_recognition moondream_object_recognition.launch.py use_remote:=false use_realsense:=True

# Set the object prompt (class to detect)
ros2 topic pub /fbot_vision/fr/object_prompt std_msgs/String "data: 'cup'"
```

### Vision Language Model

```bash
# Launch VLM service
ros2 launch fbot_vlm vlm.launch.py

# Ask questions about the current camera view (uses live camera feed)
ros2 service call /fbot_vision/vlm/question_answering/query \
    fbot_vision_msgs/srv/VLMQuestionAnswering "{question: 'What do you see?', use_image: true}"

# Ask text-only questions (no image processing)
ros2 service call /fbot_vision/vlm/question_answering/query \
    fbot_vision_msgs/srv/VLMQuestionAnswering "{question: 'What is the capital of France?', use_image: false}"

# Ask questions about a specific image (provide custom image)
ros2 service call /fbot_vision/vlm/question_answering/query \
    fbot_vision_msgs/srv/VLMQuestionAnswering "{
        question: 'Describe this image in detail', 
        use_image: true,
        image: {
            header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'camera_link'},
            height: 480, width: 640, encoding: 'rgb8',
            is_bigendian: false, step: 1920,
            data: [/* image data bytes */]
        }
    }"

# Get VLM conversation history
ros2 service call /fbot_vision/vlm/answer_history/query \
    fbot_vision_msgs/srv/VLMAnswerHistory "{questions_filter: []}"

# Get history for specific questions only
ros2 service call /fbot_vision/vlm/answer_history/query \
    fbot_vision_msgs/srv/VLMAnswerHistory "{questions_filter: ['What do you see?', 'Describe the scene']}"
```

---

## fbot_vision message and services

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/fbot_vision/fr/object_recognition` | [`Detection3DArray`](fbot_vision_msgs/msg/Detection3DArray.msg) | 3D object detections |
| `/fbot_vision/pt/tracking3D` | [`Detection3DArray`](fbot_vision_msgs/msg/Detection3DArray.msg) | 3D person tracking |
| `/fbot_vision/fr/face_recognition` | [`Detection3DArray`](fbot_vision_msgs/msg/Detection3DArray.msg) | 3D face recognition |
| `/fbot_vision/vlm/question_answering/query` | [`VLMQuestion`](fbot_vision_msgs/msg/VLMQuestion.msg) | VLM questions |
| `/fbot_vision/vlm/question_answering/answer` | [`VLMAnswer`](fbot_vision_msgs/msg/VLMAnswer.msg) | VLM responses |
| `/fbot_vision/fr/object_prompt` | `std_msgs/String` | Object prompt for Moondream |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/fbot_vision/fr/object_start` | [`std_srvs/Empty`](http://docs.ros.org/en/api/std_srvs/html/srv/Empty.html) | Start object detection |
| `/fbot_vision/fr/object_stop` | [`std_srvs/Empty`](http://docs.ros.org/en/api/std_srvs/html/srv/Empty.html) | Stop object detection |
| `/fbot_vision/pt/start` | [`std_srvs/Empty`](http://docs.ros.org/en/api/std_srvs/html/srv/Empty.html) | Start person tracking |
| `/fbot_vision/pt/stop` | [`std_srvs/Empty`](http://docs.ros.org/en/api/std_srvs/html/srv/Empty.html) | Stop person tracking |
| `/fbot_vision/vlm/question_answering/query` | [`VLMQuestionAnswering`](fbot_vision_msgs/srv/VLMQuestionAnswering.srv) | Ask VLM questions |
| `/fbot_vision/vlm/answer_history/query` | [`VLMAnswerHistory`](fbot_vision_msgs/srv/VLMAnswerHistory.srv) | Get VLM conversation history |
| `/fbot_vision/face_recognition/people_introducing` | [`PeopleIntroducing`](fbot_vision_msgs/srv/PeopleIntroducing.srv) | Register new person |
| `/fbot_vision/face_recognition/people_forgetting` | [`PeopleForgetting`](fbot_vision_msgs/srv/PeopleForgetting.srv) | Forget an existing person |
| `/fbot_vision/look_at_description` | [`LookAtDescription3D`](fbot_vision_msgs/srv/LookAtDescription3D.srv) | Look at specific 3D detection |

---

## Contributing

1. Create a feature branch (`git checkout -b feat/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to the branch (`git push origin feat/amazing-feature`)
4. Open a Pull Request