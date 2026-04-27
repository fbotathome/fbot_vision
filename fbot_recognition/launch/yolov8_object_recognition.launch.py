#!/usr/bin/env python3
from launch import LaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_remote_ssh import NodeRemoteSSH, FindPackageShareRemote
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    declared_arguments = []

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
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_femtobolt',
            default_value='false',
            description='If should launch the femtobolt'
        )
    )

    config_file_name = PythonExpression([
        "'yolov8_realsense.yaml' if '", LaunchConfiguration('use_realsense'), "' == 'true' else 'yolov8_femtobolt.yaml'"
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

    yolo_object_remote_node = NodeRemoteSSH(
        package='fbot_recognition',
        executable='yolov8_recognition',
        name=PythonExpression([
            "'yolov8_recognition_realsense' if '", LaunchConfiguration('use_realsense'), "' == 'true' else 'yolov8_recognition_femtobolt'"
        ]),
        parameters=[config_file_path_remote],
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
        name=PythonExpression([
            "'yolov8_recognition_realsense' if '", LaunchConfiguration('use_realsense'), "' == 'true' else 'yolov8_recognition_femtobolt'"
        ]),
        parameters=[config_file_path],
        condition=UnlessCondition(LaunchConfiguration('use_remote'))
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('fbot_bringup'), 'launch', 'camera.launch.py')
        ),
        launch_arguments={
            'use_realsense': LaunchConfiguration('use_realsense'),
            'use_femtobolt': LaunchConfiguration('use_femtobolt'),
        }.items()
    )

    return LaunchDescription([
        *declared_arguments,
        yolo_object_remote_node,
        yolo_object_node,
    ])