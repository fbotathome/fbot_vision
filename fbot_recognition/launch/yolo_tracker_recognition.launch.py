#!/usr/bin/env python3
from launch import LaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_remote_ssh import NodeRemoteSSH, FindPackageShareRemote
import os

def generate_launch_description():
    config_file_path_remote = PathJoinSubstitution([
        FindPackageShareRemote(remote_install_space='/home/jetson/jetson_ws/install', package='fbot_recognition'),
        'config',
        'yolo_tracker_recognition.yaml']
    )

    config_file_path = PathJoinSubstitution([
        get_package_share_directory('fbot_recognition'),
        'config',
        'yolo_tracker_recognition.yaml']
    )

    config_file_arg = DeclareLaunchArgument(
        'config',
        default_value=config_file_path,
        description='Path to the parameter file'
    )

    config_file_remote_arg = DeclareLaunchArgument(
        'remote_config',
        default_value=config_file_path_remote,
        description='Path to the parameter file'
    )

    config_remote_arg = DeclareLaunchArgument(
        'use_remote',
        default_value='true',
        description="If should run the node on remote"
    )

    yolo_tracker_remote_node = NodeRemoteSSH(
        package='fbot_recognition',
        executable='yolo_tracker_recognition',
        name='yolo_tracker_recognition',
        parameters=[LaunchConfiguration('remote_config'),],
        user='jetson',
        machine="jetson",
        source_paths=[
            "/home/jetson/jetson_ws/install/setup.bash"
        ],
        condition=IfCondition(LaunchConfiguration('use_remote'))
    )

    yolo_tracker_node = Node(
        package='fbot_recognition',
        executable='yolo_tracker_recognition',
        name='yolo_tracker_recognition',
        parameters=[LaunchConfiguration('config'),],
        user='jetson',
        machine="jetson",
        source_paths=[
            "/home/jetson/jetson_ws/install/setup.bash",
        ],
        condition=UnlessCondition(LaunchConfiguration('use_remote'))
    )

    return LaunchDescription([
        config_file_arg,
        config_file_remote_arg,
        config_remote_arg,
        yolo_tracker_remote_node,
        yolo_tracker_node
    ])
