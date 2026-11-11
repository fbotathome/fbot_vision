#!/usr/bin/env python3
"""Launches only the yolo_tracker_recognition node, locally or on the Jetson (use_remote).

The camera is not launched here; use fbot_bringup/yolo_tracker_recognition.launch.py to bring up
the camera together with the recognition.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_remote_ssh import NodeRemoteSSH, FindPackageShareRemote
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_file_path_remote = PathJoinSubstitution([
        FindPackageShareRemote(remote_install_space='/home/jetson/jetson_ws/install', package='fbot_recognition'),
        'config',
        'yolo_tracker_recognition.yaml'
    ])

    config_file_path = PathJoinSubstitution([
        get_package_share_directory('fbot_recognition'),
        'config',
        'yolo_tracker_recognition.yaml'
    ])

    declared_arguments = [
        DeclareLaunchArgument(
            'config_tracker',
            default_value=config_file_path,
            description='Path to the parameter file'
        ),
        DeclareLaunchArgument(
            'remote_config_tracker',
            default_value=config_file_path_remote,
            description='Path to the parameter file on the remote machine'
        ),
        DeclareLaunchArgument(
            'use_remote',
            default_value='true',
            description='If should run the recognition node on the remote Jetson machine'
        ),
    ]

    remote_node = NodeRemoteSSH(
        package='fbot_recognition',
        executable='yolo_tracker_recognition',
        name='yolo_tracker_recognition',
        parameters=[LaunchConfiguration('remote_config_tracker')],
        user='jetson',
        machine='jetson',
        source_paths=['/home/jetson/jetson_ws/install/setup.bash'],
        condition=IfCondition(LaunchConfiguration('use_remote'))
    )

    local_node = Node(
        package='fbot_recognition',
        executable='yolo_tracker_recognition',
        name='yolo_tracker_recognition',
        parameters=[LaunchConfiguration('config_tracker')],
        condition=UnlessCondition(LaunchConfiguration('use_remote'))
    )

    return LaunchDescription([
        *declared_arguments,
        remote_node,
        local_node,
    ])
