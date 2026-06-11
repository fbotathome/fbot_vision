#!/usr/bin/env python3

import os

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_remote_ssh import NodeRemoteSSH, FindPackageShareRemote
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    detections_topic = LaunchConfiguration("detections_topic")
    detections_topic_cmd = DeclareLaunchArgument(
        "detections_topic",
        default_value="/fbot_vision/femtobolt/object_recognition2D",
        description="Public topic for 2D detections",
    )

    detections_3d_topic = LaunchConfiguration("detections_3d_topic")
    detections_3d_topic_cmd = DeclareLaunchArgument(
        "detections_3d_topic",
        default_value="/fbot_vision/femtobolt/object_recognition",
        description="Public topic for 3D detections",
    )

    debug_topic = LaunchConfiguration("debug_topic")
    debug_topic_cmd = DeclareLaunchArgument(
        "debug_topic",
        default_value="/fbot_vision/femtobolt/object_debug",
        description="Public topic for debug image output",
    )

    config_file_path = PathJoinSubstitution([
        FindPackageShare("fbot_recognition"),
        "config",
        "yolo_ros.yaml",
    ])

    config_file_path_remote = PathJoinSubstitution([
        FindPackageShareRemote(
            remote_install_space="/home/jetson/jetson_ws/install",
            package="fbot_recognition",
        ),
        "config",
        "yolo_ros.yaml",
    ])

    config = LaunchConfiguration("config")
    config_cmd = DeclareLaunchArgument(
        "config",
        default_value=config_file_path,
        description="Path to the parameter file",
    )

    remote_config = LaunchConfiguration("remote_config")
    remote_config_cmd = DeclareLaunchArgument(
        "remote_config",
        default_value=config_file_path_remote,
        description="Path to the remote parameter file",
    )

    def run_yolo(context: LaunchContext, use_tracking, use_3d, use_remote):
        use_tracking = eval(context.perform_substitution(use_tracking))
        use_3d = eval(context.perform_substitution(use_3d))
        detections_topic_value = context.perform_substitution(detections_topic)
        detections_3d_topic_value = context.perform_substitution(detections_3d_topic)
        debug_topic_value = context.perform_substitution(debug_topic)

        new_pythonpath = os.environ.get("PYTHONPATH", "")

        input_image_topic = LaunchConfiguration("input_image_topic")
        input_image_topic_cmd = DeclareLaunchArgument(
            "input_image_topic",
            default_value="/femtobolt/color/image_raw",
            description="Name of the input image topic",
        )

        input_depth_topic = LaunchConfiguration("input_depth_topic")
        input_depth_topic_cmd = DeclareLaunchArgument(
            "input_depth_topic",
            default_value="/femtobolt/depth/image_raw",
            description="Name of the input depth topic",
        )

        input_depth_info_topic = LaunchConfiguration("input_depth_info_topic")
        input_depth_info_topic_cmd = DeclareLaunchArgument(
            "input_depth_info_topic",
            default_value="/femtobolt/depth/camera_info",
            description="Name of the input depth info topic",
        )

        use_debug = LaunchConfiguration("use_debug")
        use_debug_cmd = DeclareLaunchArgument(
            "use_debug",
            default_value="True",
            description="Whether to activate the debug node",
        )

        detect_3d_detections_topic = detections_topic_value
        debug_detections_topic = detections_topic_value

        if use_tracking:
            detect_3d_detections_topic = "tracking"

        if use_tracking and not use_3d:
            debug_detections_topic = "tracking"
        elif use_3d:
            debug_detections_topic = detections_3d_topic_value

        remote_user = "jetson"
        remote_machine = "jetson"
        remote_source_paths = ["/home/jetson/jetson_ws/install/setup.bash"]

        yolo_node_remappings = [
            ("image_raw", input_image_topic),
            ("detections", detections_topic_value),
        ]

        yolo_node_remote_cmd = NodeRemoteSSH(
            package="fbot_recognition",
            executable="yolo_node",
            name="yolo_node",
            parameters=[remote_config],
            remappings=yolo_node_remappings,
            user=remote_user,
            machine=remote_machine,
            source_paths=remote_source_paths,
            condition=IfCondition(use_remote),
        )

        yolo_node_cmd = Node(
            package="fbot_recognition",
            executable="yolo_node",
            name="yolo_node",
            additional_env={"PYTHONPATH": new_pythonpath},
            parameters=[config],
            remappings=yolo_node_remappings,
            condition=UnlessCondition(use_remote),
        )

        tracking_node_remappings = [
            ("image_raw", input_image_topic),
            ("detections", detections_topic_value),
        ]

        tracking_node_remote_cmd = NodeRemoteSSH(
            package="fbot_recognition",
            executable="tracking_node",
            name="tracking_node",
            parameters=[remote_config],
            remappings=tracking_node_remappings,
            user=remote_user,
            machine=remote_machine,
            source_paths=remote_source_paths,
            condition=IfCondition(PythonExpression(["'", use_remote, "'.lower() in ('true', '1') and ", str(use_tracking)])),
        )

        tracking_node_cmd = Node(
            package="fbot_recognition",
            executable="tracking_node",
            name="tracking_node",
            additional_env={"PYTHONPATH": new_pythonpath},
            parameters=[config],
            remappings=tracking_node_remappings,
            condition=IfCondition(PythonExpression(["'", use_remote, "'.lower() not in ('true', '1') and ", str(use_tracking)])),
        )

        detect_3d_node_remappings = [
            ("depth_image", input_depth_topic),
            ("depth_info", input_depth_info_topic),
            ("detections", detect_3d_detections_topic),
            ("detections_3d", detections_3d_topic_value),
        ]

        detect_3d_node_remote_cmd = NodeRemoteSSH(
            package="fbot_recognition",
            executable="detect_3d_node",
            name="detect_3d_node",
            parameters=[remote_config],
            remappings=detect_3d_node_remappings,
            user=remote_user,
            machine=remote_machine,
            source_paths=remote_source_paths,
            condition=IfCondition(PythonExpression(["'", use_remote, "'.lower() in ('true', '1') and ", str(use_3d)])),
        )

        detect_3d_node_cmd = Node(
            package="fbot_recognition",
            executable="detect_3d_node",
            name="detect_3d_node",
            additional_env={"PYTHONPATH": new_pythonpath},
            parameters=[config],
            remappings=detect_3d_node_remappings,
            condition=IfCondition(PythonExpression(["'", use_remote, "'.lower() not in ('true', '1') and ", str(use_3d)])),
        )

        debug_node_remappings = [
            ("image_raw", input_image_topic),
            ("detections", debug_detections_topic),
            ("dbg_image", debug_topic_value),
        ]

        debug_node_remote_cmd = NodeRemoteSSH(
            package="fbot_recognition",
            executable="debug_node",
            name="debug_node",
            parameters=[remote_config],
            remappings=debug_node_remappings,
            user=remote_user,
            machine=remote_machine,
            source_paths=remote_source_paths,
            condition=IfCondition(PythonExpression(["'", use_remote, "'.lower() in ('true', '1') and ", use_debug])),
        )

        debug_node_cmd = Node(
            package="fbot_recognition",
            executable="debug_node",
            name="debug_node",
            additional_env={"PYTHONPATH": new_pythonpath},
            parameters=[config],
            remappings=debug_node_remappings,
            condition=IfCondition(PythonExpression(["'", use_remote, "'.lower() not in ('true', '1') and ", use_debug])),
        )

        return (
            input_image_topic_cmd,
            input_depth_topic_cmd,
            input_depth_info_topic_cmd,
            use_debug_cmd,
            detections_topic_cmd,
            detections_3d_topic_cmd,
            debug_topic_cmd,
            yolo_node_remote_cmd,
            yolo_node_cmd,
            tracking_node_remote_cmd,
            tracking_node_cmd,
            detect_3d_node_remote_cmd,
            detect_3d_node_cmd,
            debug_node_remote_cmd,
            debug_node_cmd,
        )

    use_tracking = LaunchConfiguration("use_tracking")
    use_tracking_cmd = DeclareLaunchArgument(
        "use_tracking",
        default_value="False",
        description="Whether to activate tracking",
    )

    use_3d = LaunchConfiguration("use_3d")
    use_3d_cmd = DeclareLaunchArgument(
        "use_3d",
        default_value="True",
        description="Whether to activate 3D detections",
    )

    use_remote = LaunchConfiguration("use_remote")
    use_remote_cmd = DeclareLaunchArgument(
        "use_remote",
        default_value="true",
        description="If it should run the nodes on the remote machine (jetson) via SSH",
    )

    use_realsense = LaunchConfiguration("use_realsense")
    use_realsense_cmd = DeclareLaunchArgument(
        "use_realsense",
        default_value="false",
        description="If it should run the realsense node",
    )

    use_femtobolt = LaunchConfiguration("use_femtobolt")
    use_femtobolt_cmd = DeclareLaunchArgument(
        "use_femtobolt",
        default_value="false",
        description="If it should launch the femtobolt",
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("fbot_bringup"), "launch", "camera.launch.py"
            )
        ),
        launch_arguments={
            "use_realsense": use_realsense,
            "use_femtobolt": use_femtobolt,
            "use_remote": use_remote,
            "validate_config": "false",
        }.items(),
    )

    return LaunchDescription(
        [
            config_cmd,
            remote_config_cmd,
            detections_topic_cmd,
            detections_3d_topic_cmd,
            debug_topic_cmd,
            use_tracking_cmd,
            use_3d_cmd,
            use_remote_cmd,
            use_realsense_cmd,
            use_femtobolt_cmd,
            OpaqueFunction(function=run_yolo, args=[use_tracking, use_3d, use_remote]),
            camera,
        ]
    )
