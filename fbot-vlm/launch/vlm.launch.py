import os
import launch
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('fbot_vlm'),
        'config',
        'vision_language_model.yaml'
        )

        
    return launch.LaunchDescription([
        Node(
            package='fbot_vlm',
            executable='vision_language_model',
            name='vision_language_model',
            parameters = [config])
    ])