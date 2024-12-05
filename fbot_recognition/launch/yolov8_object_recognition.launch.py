#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

fbot_recognition_params = [{'name': 'config_file', 'default': 'yolov8_object_recognition.yaml', 'description': 'Path to the parameter file'},
                           {'name': 'threshold', 'default': '0.5', 'description': 'Confidence threshold for detection'},
                           {'name': 'classes_by_category', 'default': '', 'description': 'Detection classes by category'},
                           {'name': 'max_sizes', 'default': '[0.5, 0.5, 0.5]', 'description': 'Maximum sizes for detection'},
                           {'name': 'model_file', 'default': 'yolov8n.pt', 'description': 'Name of the model file'},
                           ]

def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(param['name'], default_value=param['default'], description=param['description']) for param in parameters]

def set_configurable_parameters(parameters):
    return dict([(param['name'], LaunchConfiguration(param['name'])) for param in parameters])

def launch_setup(context, params):

    dynamic_config_path = PathJoinSubstitution(
        substitutions=[
            FindPackageShare('fbot_recognition'), 'config', LaunchConfiguration('config_file')
        ]
    )

    return [Node(
        package='fbot_recognition',
        executable='yolov8_recognition',
        name='yolov8_recognition',
        parameters=[dynamic_config_path]
    )]

def generate_launch_description():
    return LaunchDescription( declare_configurable_parameters(fbot_recognition_params) + [
        OpaqueFunction(function=launch_setup, kwargs={'params': set_configurable_parameters(fbot_recognition_params)})
    ])