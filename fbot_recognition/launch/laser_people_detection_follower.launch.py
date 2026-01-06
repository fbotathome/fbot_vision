import launch
import launch_ros.actions
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch the complete laser-based person detection and following system."""
    return launch.LaunchDescription([
        # Person detection node
        launch_ros.actions.Node(
            package='fbot_recognition',
            executable='lasermodelnode',
            name='laser_model_host',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('fbot_recognition'),
                    'config',
                    'laser_model_params.yaml'
                ])
            ],
            output='screen'
        ),

        # Person follower node
        launch_ros.actions.Node(
            package='fbot_recognition',
            executable='lasermodelnode_follower',
            name='laser_people_follower',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('fbot_recognition'),
                    'config',
                    'laser_model_params.yaml'
                ])
            ],
            output='screen'
        )
    ])