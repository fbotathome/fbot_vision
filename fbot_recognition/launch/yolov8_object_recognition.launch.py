#!/usr/bin/env python3
"""Launches only the yolov8_recognition node(s), locally or on the Jetson (use_remote).

One node per selected camera, each with its own config (topics):
    use_femtobolt -> yolov8_recognition_femtobolt (yolov8_femtobolt.yaml)
    use_realsense -> yolov8_recognition_realsense (yolov8_realsense.yaml)
With both true, both nodes run. With both false, the Femto Bolt one runs (default camera).

The camera itself is not launched here; use fbot_bringup/yolov8_object_recognition.launch.py
(use_camera:=true) to bring up the camera together with the recognition.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_remote_ssh import NodeRemoteSSH, FindPackageShareRemote


def recognition_nodes(camera, enabled):
    """Local and remote yolov8_recognition node for one camera, run when `enabled` is 'true'."""
    config_file = f'yolov8_{camera}.yaml'
    node_name = f'yolov8_recognition_{camera}'

    def when(remote):
        return IfCondition(PythonExpression([
            "'true' if (", *enabled, ") and '", LaunchConfiguration('use_remote'), "' == '", remote, "' else 'false'"
        ]))

    remote_node = NodeRemoteSSH(
        package='fbot_recognition',
        executable='yolov8_recognition',
        name=node_name,
        parameters=[PathJoinSubstitution([
            FindPackageShareRemote(remote_install_space='/home/jetson/jetson_ws/install', package='fbot_recognition'),
            'config',
            config_file
        ])],
        user='jetson',
        machine='jetson',
        source_paths=['/home/jetson/jetson_ws/install/setup.bash'],
        condition=when('true')
    )

    local_node = Node(
        package='fbot_recognition',
        executable='yolov8_recognition',
        name=node_name,
        parameters=[PathJoinSubstitution([FindPackageShare('fbot_recognition'), 'config', config_file])],
        condition=when('false')
    )

    return [remote_node, local_node]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            'use_remote',
            default_value='true',
            description='If should run the recognition node(s) on the remote Jetson machine'
        ),
        DeclareLaunchArgument(
            'use_femtobolt',
            default_value='false',
            description='Run the recognition on the Femto Bolt topics (also the default when no camera is selected). '
                        'Does not launch the camera'
        ),
        DeclareLaunchArgument(
            'use_realsense',
            default_value='false',
            description='Run the recognition on the RealSense topics. Does not launch the camera'
        ),
    ]

    femtobolt = LaunchConfiguration('use_femtobolt')
    realsense = LaunchConfiguration('use_realsense')

    # Femto Bolt when selected, or when no camera is selected (default)
    femtobolt_enabled = ["'", femtobolt, "' == 'true' or '", realsense, "' != 'true'"]
    realsense_enabled = ["'", realsense, "' == 'true'"]

    return LaunchDescription([
        *declared_arguments,
        *recognition_nodes('femtobolt', femtobolt_enabled),
        *recognition_nodes('realsense', realsense_enabled),
    ])
