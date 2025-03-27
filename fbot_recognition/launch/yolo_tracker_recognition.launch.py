#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_file_path = os.path.join(
        get_package_share_directory('fbot_recognition'),
        'config',
        'yolo_tracker_recognition.yaml'
    )

    config_file_arg = DeclareLaunchArgument(
        'config',
        default_value=config_file_path,
        description='Path to the parameter file'
    )

    yolo_tracker_node = Node(
        package='fbot_recognition',
        executable='yolo_tracker_recognition',
        name='yolo_tracker_recognition',
        parameters=[LaunchConfiguration('config')]
    )

    return LaunchDescription([
        config_file_arg,
        yolo_tracker_node
    ])