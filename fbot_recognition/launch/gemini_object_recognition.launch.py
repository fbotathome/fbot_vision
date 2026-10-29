#!/usr/bin/env python3
from launch import LaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_remote_ssh import NodeRemoteSSH, FindPackageShareRemote
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    config_file_path_remote = PathJoinSubstitution([
        FindPackageShareRemote(remote_install_space='/home/jetson/jetson_ws/install', package='fbot_recognition'),
        'config',
        'gemini_object_recognition.yaml']
    )

    config_file_path = PathJoinSubstitution([
        get_package_share_directory('fbot_recognition'),
        'config',
        'gemini_object_recognition.yaml']
    )

    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'config',
            default_value=config_file_path,
            description='Path to the parameter file'
        ))
    declared_arguments.append(
        DeclareLaunchArgument(
            'remote_config',
            default_value=config_file_path_remote,
            description='Path to the remote parameter file'
        ))
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_remote',
            default_value='false',
            description="If it should run the node on remote"
        ))

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_realsense',
            default_value='false',
            description="If it should run the realsense node"
        ))
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_femtobolt',
            default_value='false',
            description='If should launch the femtobolt'
        )
    )

    gemini_object_remote_node = NodeRemoteSSH(
        package='fbot_recognition',
        executable='gemini_recognition',
        name='gemini_recognition',
        parameters=[LaunchConfiguration('remote_config'),],
        user='jetson',
        machine="jetson",
        source_paths=[
            "/home/jetson/jetson_ws/install/setup.bash"
        ],
        condition=IfCondition(LaunchConfiguration('use_remote'))
    )

    gemini_object_node = Node(
        package='fbot_recognition',
        executable='gemini_recognition',
        name='gemini_recognition',
        parameters=[LaunchConfiguration('config'),],
        condition=UnlessCondition(LaunchConfiguration('use_remote'))
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('fbot_bringup'), 'launch', 'camera.launch.py')
        ),
        launch_arguments={
            'use_realsense': LaunchConfiguration('use_realsense'),
            'use_femtobolt': LaunchConfiguration('use_femtobolt'),
            'use_remote': LaunchConfiguration('use_remote'),
            'validate_config': 'false'
        }.items()
    )

    return LaunchDescription([
        *declared_arguments,
        gemini_object_remote_node,
        gemini_object_node,
        # camera
    ])
