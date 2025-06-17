#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition, UnlessCondition
from launch_remote_ssh import NodeRemoteSSH, FindPackageShareRemote
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():

    config_file_path_remote = PathJoinSubstitution([
        FindPackageShareRemote(remote_install_space='/home/jetson/jetson_ws/install', package='fbot_recognition'),
        'config',
        'face_recognition.yaml']
    )

    config_file_path = PathJoinSubstitution([
        get_package_share_directory('fbot_recognition'),
        'config',
        'face_recognition.yaml']
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
            default_value='true',
            description="If it should run the node on remote"
        ))
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_realsense',
            default_value='false',
            description="If it should run the realsense node"
        ))


    face_recognition_remote_node = NodeRemoteSSH(
        package='fbot_recognition',
        executable='face_recognition',
        name='face_recognition',
        parameters=[LaunchConfiguration('remote_config'),],
        user='jetson',
        machine="jetson",
        source_paths=[
            "/home/jetson/jetson_ws/install/setup.bash"
        ],
        condition=IfCondition(LaunchConfiguration('use_remote'))
    )

    face_recognition_node = Node(
        package='fbot_recognition',
        executable='face_recognition',
        name='face_recognition',
        parameters=[LaunchConfiguration('config'),],
        condition=UnlessCondition(LaunchConfiguration('use_remote'))
    )

    realsense2_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'camera_name': 'realsense',
            'camera_namespace': 'fbot_vision',
            'enable_rgbd': 'true',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': 'true'
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_realsense'))
    )


    return LaunchDescription([
        *declared_arguments,
        face_recognition_remote_node,
        face_recognition_node,
        realsense2_node
    ])