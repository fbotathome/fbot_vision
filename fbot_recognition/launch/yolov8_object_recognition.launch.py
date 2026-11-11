#!/usr/bin/env python3
"""Launches only the yolov8_recognition node, locally or on the Jetson (use_remote).

The camera is not launched here; use_realsense only selects which camera config
(topics) the node uses. Use fbot_bringup/yolov8_object_recognition.launch.py to
bring up the camera together with the recognition.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_remote_ssh import NodeRemoteSSH, FindPackageShareRemote


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            'use_remote',
            default_value='true',
            description='If should run the recognition node on the remote Jetson machine'
        ),
        DeclareLaunchArgument(
            'use_realsense',
            default_value='false',
            description='Use the RealSense config (topics) instead of the Femto Bolt one. Does not launch the camera'
        ),
    ]

    config_file_name = PythonExpression([
        "'yolov8_realsense.yaml' if '", LaunchConfiguration('use_realsense'), "' == 'true' else 'yolov8_femtobolt.yaml'"
    ])

    node_name = PythonExpression([
        "'yolov8_recognition_realsense' if '", LaunchConfiguration('use_realsense'), "' == 'true' else 'yolov8_recognition_femtobolt'"
    ])

    config_file_path = PathJoinSubstitution([
        FindPackageShare('fbot_recognition'),
        'config',
        config_file_name
    ])

    config_file_path_remote = PathJoinSubstitution([
        FindPackageShareRemote(remote_install_space='/home/jetson/jetson_ws/install', package='fbot_recognition'),
        'config',
        config_file_name
    ])

    remote_node = NodeRemoteSSH(
        package='fbot_recognition',
        executable='yolov8_recognition',
        name=node_name,
        parameters=[config_file_path_remote],
        user='jetson',
        machine='jetson',
        source_paths=['/home/jetson/jetson_ws/install/setup.bash'],
        condition=IfCondition(LaunchConfiguration('use_remote'))
    )

    local_node = Node(
        package='fbot_recognition',
        executable='yolov8_recognition',
        name=node_name,
        parameters=[config_file_path],
        condition=UnlessCondition(LaunchConfiguration('use_remote'))
    )

    return LaunchDescription([
        *declared_arguments,
        remote_node,
        local_node,
    ])
