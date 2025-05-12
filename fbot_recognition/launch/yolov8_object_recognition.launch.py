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
        'yolov8_object_recognition.yaml']
    )

    config_file_path = PathJoinSubstitution([
        get_package_share_directory('fbot_recognition'),
        'config',
        'yolov8_object_recognition.yaml']
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

    use_realsense_arg = DeclareLaunchArgument(
        'use_realsense',
        default_value='false',
        description="If should launch the camera node"
    )

    yolo_object_remote_node = NodeRemoteSSH(
        package='fbot_recognition',
        executable='yolov8_recognition',
        name='yolov8_recognition',
        parameters=[LaunchConfiguration('remote_config'),],
        user='jetson',
        machine="jetson",
        source_paths=[
            "/home/jetson/jetson_ws/install/setup.bash"
        ],
        condition=IfCondition(LaunchConfiguration('use_remote'))
    )

    yolo_object_node = Node(
        package='fbot_recognition',
        executable='yolov8_recognition',
        name='yolov8_recognition',
        parameters=[LaunchConfiguration('config'),],
        condition=UnlessCondition(LaunchConfiguration('use_remote'))
    )


    realsense2_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ),
        launch_arguments={
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
        config_file_arg,
        config_file_remote_arg,
        config_remote_arg,
        use_realsense_arg,
        yolo_object_remote_node,
        yolo_object_node,
        realsense2_node
    ])