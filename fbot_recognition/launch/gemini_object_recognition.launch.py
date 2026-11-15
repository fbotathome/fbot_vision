#!/usr/bin/env python3
"""Launches only the gemini_recognition node, locally or on the Jetson (use_remote).

The camera is not launched here; use fbot_bringup/gemini_object_recognition.launch.py to bring up
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
        'gemini_object_recognition.yaml'
    ])

    config_file_path = PathJoinSubstitution([
        get_package_share_directory('fbot_recognition'),
        'config',
        'gemini_object_recognition.yaml'
    ])

    declared_arguments = [
        DeclareLaunchArgument(
            'config',
            default_value=config_file_path,
            description='Path to the parameter file'
        ),
        DeclareLaunchArgument(
            'remote_config',
            default_value=config_file_path_remote,
            description='Path to the parameter file on the remote machine'
        ),
        DeclareLaunchArgument(
            'use_remote',
            default_value='false',
            description='If should run the recognition node on the remote Jetson machine'
        ),
    ]

    remote_node = NodeRemoteSSH(
        package='fbot_recognition',
        executable='gemini_recognition',
        name='gemini_recognition',
        parameters=[LaunchConfiguration('remote_config')],
        user='jetson',
        machine='jetson',
        source_paths=['/home/jetson/jetson_ws/install/setup.bash'],
        condition=IfCondition(LaunchConfiguration('use_remote'))
    )

    local_node = Node(
        package='fbot_recognition',
        executable='gemini_recognition',
        name='gemini_recognition',
        parameters=[LaunchConfiguration('config')],
        condition=UnlessCondition(LaunchConfiguration('use_remote'))
    )

    return LaunchDescription([
        *declared_arguments,
        remote_node,
        local_node,
    ])
