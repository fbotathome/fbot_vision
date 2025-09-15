#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('fbot_recognition')
    yolo_tracker_cfg = os.path.join(pkg_share, 'config', 'yolo_tracker_config', 'yolo_tracker_default_config.yaml')

    return LaunchDescription([
        Node(
            package='fbot_recognition',
            executable='camera_lidar_person_tracker',
            name='camera_lidar_person_tracker',
            output='screen',
            parameters=[
                {'tracking.config_file': 'yolo_tracker_default_config.yaml'},
                {'model_file': 'yolo11n-pose'},
                {'fusion.lidar.min_points': 15},
                {'fusion.priority': 'lidar'},
                {'fusion.lidar.use': True},
            ]
        )
    ])
