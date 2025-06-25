#!/usr/bin/env python3
import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch_remote_ssh import NodeRemoteSSH, FindPackageShareRemote
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

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

    launch_realsense_arg = DeclareLaunchArgument(
        'use_realsense',
        default_value='true',
        description="If should launch the camera node"
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
        condition=UnlessCondition(LaunchConfiguration('use_remote'))
    )

    realsense2_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'camera_name': 'camera',
            'camera_namespace': 'fbot_vision',
            'enable_rgbd': 'true',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': 'true'
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_realsense')))

    return LaunchDescription([
        config_file_arg,
        config_file_remote_arg,
        config_remote_arg,
        yolo_tracker_remote_node,
        yolo_tracker_node,
        launch_realsense_arg,
        realsense2_node
    ])
