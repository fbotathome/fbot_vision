# FBot Recognition Package

This package provides computer vision and sensor-based recognition capabilities for the FBot robot platform.

## Components

### Laser-Based Person Detection and Tracking

This package includes a complete laser-based person detection and following system using deep learning.

#### Nodes

1. **LaserModelHost** (`lasermodelnode`)
   - Runs ONNX-based person detection model on laser scan data
   - Publishes detected people as `PersonDetectionList` messages
   - Publishes visualization markers for detected people

2. **LaserModelHostFollower** (`lasermodelnode_follower`)
   - Subscribes to person detections
   - Tracks people across frames using advanced data association
   - Controls robot velocity to follow selected person
   - Publishes tracking visualization markers

#### Topics

- **Published:**
  - `detected_people` (`fbot_vision_msgs/PersonDetectionList`): Detected people
  - `detected_people_markers` (`visualization_msgs/MarkerArray`): Visualization markers
  - `tracked_people_markers` (`visualization_msgs/MarkerArray`): Tracking markers with confidence
  - `/cmd_vel` (`geometry_msgs/Twist`): Velocity commands for following

- **Subscribed:**
  - `/scan` (`sensor_msgs/LaserScan`): Laser scan data
  - `detected_people` (`fbot_vision_msgs/PersonDetectionList`): Person detections
  - `select_person` (`std_msgs/Int32`): Person selection commands (-1 for auto-select)

#### Parameters

See `config/laser_model_params.yaml` for all configuration parameters.

Key parameters:
- `model_file`: Path to ONNX model file
- `max_linear_velocity`: Maximum linear velocity for following (m/s)
- `max_angular_velocity`: Maximum angular velocity for following (rad/s)
- `stopping_distance`: Distance to maintain from target (m)
- `max_tracking_distance`: Maximum distance for associating detections (m)
- `confidence_threshold`: Minimum confidence for valid tracks

#### Launch Files

- `laser_model_host.launch.py`: Launches only the detection node
- `laser_people_follower.launch.py`: Launches only the follower node
- `laser_people_detection_follower.launch.py`: Launches the complete system

#### Usage

```bash
# Launch complete system
ros2 launch fbot_recognition laser_people_detection_follower.launch.py

# Select person to follow (use -1 for auto-select)
ros2 topic pub /select_person std_msgs/Int32 "data: 1"
```

### Other Recognition Modules

This package also includes other recognition modules (face detection, object detection, etc.) in separate subdirectories.