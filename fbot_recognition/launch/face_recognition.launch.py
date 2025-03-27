#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_file_path = os.path.join(
        get_package_share_directory('fbot_recognition'),
        'config',
        'face_recognition.yaml'
    )

    config_file_arg = DeclareLaunchArgument(
        'config',
        default_value=config_file_path,
        description='Path to the parameter file'
    )

    face_recog_node = Node(
        package='fbot_recognition',
        executable='face_recognition',
        name='face_recognition',
        parameters=[LaunchConfiguration('config')]
    )

    return LaunchDescription([
        config_file_arg,
        yolov8_recog_node
    ])