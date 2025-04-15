#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_remote_ssh import NodeRemoteSSH, FindPackageShareRemote
import os

def generate_launch_description():
    config_file_path = PathJoinSubstitution([
        FindPackageShareRemote(remote_install_space='/home/jetson/jetson_ws/install', package='fbot_recognition'),
        'config',
        'yolo_tracker_recognition.yaml']
    )

    config_file_arg = DeclareLaunchArgument(
        'config',
        default_value=config_file_path,
        description='Path to the parameter file'
    )

    yolo_tracker_node = NodeRemoteSSH(
        package='fbot_recognition',
        executable='yolo_tracker_recognition',
        name='yolo_tracker_recognition',
        parameters=[LaunchConfiguration('config'),],
        user='jetson',
        machine="jetson",
        source_paths=[
            "/home/jetson/jetson_ws/install/setup.bash",
        ]
    )

    return LaunchDescription([
        config_file_arg,
        yolo_tracker_node
    ])