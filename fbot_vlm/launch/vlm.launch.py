import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    config_file_path = os.path.join(
        get_package_share_directory('fbot_vlm'),
        'config',
        'quiz_config.yaml'
        )

    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'config_vlm',
            default_value=config_file_path,
            description='Path to the parameter file'
        ))

    declared_arguments.append(
        DeclareLaunchArgument(
            'vlm_api_type',
            default_value='ollama',
            description='Type of the VLM API. Must be one of: [openai, ollama, google-genai]'
        ))

    declared_arguments.append(
        DeclareLaunchArgument(
            'vlm_api_model',
            default_value='gemma3:4b',
            description='Model to use for the VLM API. Example: gemma3:4b, gemini-1.5-pro:8b, etc.'
        ))

    declared_arguments.append(
        DeclareLaunchArgument(
            'vlm_api_host',
            default_value='http://192.168.1.189:11434',
            description='Host URL for the VLM API. Must be set for openai and ollama.'
        ))

    vlm_node = Node(
        package='fbot_vlm',
        executable='vision_language_model',
        name='vision_language_model',
        namespace='fbot_vision/vlm',
        parameters=[
            LaunchConfiguration('config_vlm'),
            {'vlm_api_type': LaunchConfiguration('vlm_api_type')},
            {'vlm_api_model': LaunchConfiguration('vlm_api_model')},
            {'vlm_api_host': LaunchConfiguration('vlm_api_host')},
            ],
    )

        
    return LaunchDescription([
        *declared_arguments,
        vlm_node,

    ])