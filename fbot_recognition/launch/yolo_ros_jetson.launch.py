#!/usr/bin/env python3

import os

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():

    detections_topic = LaunchConfiguration("detections_topic")
    detections_topic_cmd = DeclareLaunchArgument(
        "detections_topic",
        default_value="/fbot_vision/fr/object_recognition2D",
        description="Public topic for 2D detections",
    )

    detections_3d_topic = LaunchConfiguration("detections_3d_topic")
    detections_3d_topic_cmd = DeclareLaunchArgument(
        "detections_3d_topic",
        default_value="/fbot_vision/fr/object_recognition",
        description="Public topic for 3D detections",
    )

    debug_topic = LaunchConfiguration("debug_topic")
    debug_topic_cmd = DeclareLaunchArgument(
        "debug_topic",
        default_value="/fbot_vision/fr/object_debug",
        description="Public topic for debug image output",
    )

    def run_yolo(context: LaunchContext, use_tracking, use_3d):
        use_tracking = eval(context.perform_substitution(use_tracking))
        use_3d = eval(context.perform_substitution(use_3d))
        detections_topic_value = context.perform_substitution(detections_topic)
        detections_3d_topic_value = context.perform_substitution(detections_3d_topic)
        debug_topic_value = context.perform_substitution(debug_topic)

        new_pythonpath = os.environ.get("PYTHONPATH", "")

        model = LaunchConfiguration("model")
        model_cmd = DeclareLaunchArgument(
            "model",
            default_value="yolov8m.pt",
            description="Model name or path",
        )

        tracker = LaunchConfiguration("tracker")
        tracker_cmd = DeclareLaunchArgument(
            "tracker",
            default_value="bytetrack.yaml",
            description="Tracker name or path",
        )

        device = LaunchConfiguration("device")
        device_cmd = DeclareLaunchArgument(
            "device",
            default_value="cuda:0",
            description="Device to use (GPU/CPU)",
        )

        fuse_model = LaunchConfiguration("fuse_model")
        fuse_model_cmd = DeclareLaunchArgument(
            "fuse_model",
            default_value="True",
            description="Whether to fuse the model for inference optimization",
        )

        yolo_encoding = LaunchConfiguration("yolo_encoding")
        yolo_encoding_cmd = DeclareLaunchArgument(
            "yolo_encoding",
            default_value="bgr8",
            description="Encoding of the input image topic",
        )

        enable = LaunchConfiguration("enable")
        enable_cmd = DeclareLaunchArgument(
            "enable",
            default_value="True",
            description="Whether to start YOLO enabled",
        )

        threshold = LaunchConfiguration("threshold")
        threshold_cmd = DeclareLaunchArgument(
            "threshold",
            default_value="0.5",
            description="Minimum probability of a detection to be published",
        )

        iou = LaunchConfiguration("iou")
        iou_cmd = DeclareLaunchArgument(
            "iou",
            default_value="0.7",
            description="IoU threshold",
        )

        imgsz_height = LaunchConfiguration("imgsz_height")
        imgsz_height_cmd = DeclareLaunchArgument(
            "imgsz_height",
            default_value="480",
            description="Image height for inference",
        )

        imgsz_width = LaunchConfiguration("imgsz_width")
        imgsz_width_cmd = DeclareLaunchArgument(
            "imgsz_width",
            default_value="640",
            description="Image width for inference",
        )

        half = LaunchConfiguration("half")
        half_cmd = DeclareLaunchArgument(
            "half",
            default_value="True",
            description="Whether to enable half-precision inference",
        )

        max_det = LaunchConfiguration("max_det")
        max_det_cmd = DeclareLaunchArgument(
            "max_det",
            default_value="300",
            description="Maximum number of detections allowed per image",
        )

        augment = LaunchConfiguration("augment")
        augment_cmd = DeclareLaunchArgument(
            "augment",
            default_value="False",
            description="Whether to enable test-time augmentation",
        )

        agnostic_nms = LaunchConfiguration("agnostic_nms")
        agnostic_nms_cmd = DeclareLaunchArgument(
            "agnostic_nms",
            default_value="False",
            description="Whether to enable class-agnostic NMS",
        )

        retina_masks = LaunchConfiguration("retina_masks")
        retina_masks_cmd = DeclareLaunchArgument(
            "retina_masks",
            default_value="False",
            description="Whether to use high-resolution segmentation masks",
        )

        input_image_topic = LaunchConfiguration("input_image_topic")
        input_image_topic_cmd = DeclareLaunchArgument(
            "input_image_topic",
            default_value="/femtobolt/color/image_raw",
            description="Name of the input image topic",
        )

        image_reliability = LaunchConfiguration("image_reliability")
        image_reliability_cmd = DeclareLaunchArgument(
            "image_reliability",
            default_value="1",
            choices=["0", "1", "2"],
            description="QoS reliability of the input image topic",
        )

        input_depth_topic = LaunchConfiguration("input_depth_topic")
        input_depth_topic_cmd = DeclareLaunchArgument(
            "input_depth_topic",
            default_value="/femtobolt/depth/image_raw",
            description="Name of the input depth topic",
        )

        depth_image_reliability = LaunchConfiguration("depth_image_reliability")
        depth_image_reliability_cmd = DeclareLaunchArgument(
            "depth_image_reliability",
            default_value="1",
            choices=["0", "1", "2"],
            description="QoS reliability of the input depth image topic",
        )

        input_depth_info_topic = LaunchConfiguration("input_depth_info_topic")
        input_depth_info_topic_cmd = DeclareLaunchArgument(
            "input_depth_info_topic",
            default_value="/femtobolt/depth/camera_info",
            description="Name of the input depth info topic",
        )

        depth_info_reliability = LaunchConfiguration("depth_info_reliability")
        depth_info_reliability_cmd = DeclareLaunchArgument(
            "depth_info_reliability",
            default_value="1",
            choices=["0", "1", "2"],
            description="QoS reliability of the input depth info topic",
        )

        target_frame = LaunchConfiguration("target_frame")
        target_frame_cmd = DeclareLaunchArgument(
            "target_frame",
            default_value="base_link",
            description="Target frame to transform the 3D boxes",
        )

        depth_image_units_divisor = LaunchConfiguration("depth_image_units_divisor")
        depth_image_units_divisor_cmd = DeclareLaunchArgument(
            "depth_image_units_divisor",
            default_value="1000",
            description="Divisor used to convert raw depth values into metres",
        )

        enable_foreground_filter = LaunchConfiguration("enable_foreground_filter")
        enable_foreground_filter_cmd = DeclareLaunchArgument(
            "enable_foreground_filter",
            default_value="True",
            description="Enable mask erosion and foreground depth filtering",
        )

        enable_bimodal_filter = LaunchConfiguration("enable_bimodal_filter")
        enable_bimodal_filter_cmd = DeclareLaunchArgument(
            "enable_bimodal_filter",
            default_value="True",
            description="Enable bimodal depth background suppression",
        )

        mask_erosion_ratio = LaunchConfiguration("mask_erosion_ratio")
        mask_erosion_ratio_cmd = DeclareLaunchArgument(
            "mask_erosion_ratio",
            default_value="0.08",
            description="Fraction of the mask size used for erosion",
        )

        bimodal_min_gap = LaunchConfiguration("bimodal_min_gap")
        bimodal_min_gap_cmd = DeclareLaunchArgument(
            "bimodal_min_gap",
            default_value="0.08",
            description="Minimum depth gap to consider two peaks as bimodal",
        )

        max_depth_extent_default = LaunchConfiguration("max_depth_extent_default")
        max_depth_extent_default_cmd = DeclareLaunchArgument(
            "max_depth_extent_default",
            default_value="0.60",
            description="Maximum expected depth extent for unknown classes",
        )

        max_depth_extent_bottle = LaunchConfiguration("max_depth_extent_bottle")
        max_depth_extent_bottle_cmd = DeclareLaunchArgument(
            "max_depth_extent_bottle",
            default_value="0.15",
            description="Maximum expected depth extent for bottles",
        )

        namespace = LaunchConfiguration("namespace")
        namespace_cmd = DeclareLaunchArgument(
            "namespace",
            default_value="yolo",
            description="Namespace for the nodes",
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

        yolo_node_cmd = Node(
            package="fbot_recognition",
            executable="yolo_node",
            name="yolo_node",
            namespace=namespace,
            additional_env={"PYTHONPATH": new_pythonpath},
            parameters=[
                {
                    "model": model,
                    "device": device,
                    "fuse_model": fuse_model,
                    "yolo_encoding": yolo_encoding,
                    "enable": enable,
                    "threshold": threshold,
                    "iou": iou,
                    "imgsz_height": imgsz_height,
                    "imgsz_width": imgsz_width,
                    "half": half,
                    "max_det": max_det,
                    "augment": augment,
                    "agnostic_nms": agnostic_nms,
                    "retina_masks": retina_masks,
                    "image_reliability": image_reliability,
                }
            ],
            remappings=[
                ("image_raw", input_image_topic),
                ("detections", detections_topic_value),
            ],
        )

        tracking_node_cmd = Node(
            package="fbot_recognition",
            executable="tracking_node",
            name="tracking_node",
            namespace=namespace,
            additional_env={"PYTHONPATH": new_pythonpath},
            parameters=[{"tracker": tracker, "image_reliability": image_reliability}],
            remappings=[
                ("image_raw", input_image_topic),
                ("detections", detections_topic_value),
            ],
            condition=IfCondition(PythonExpression([str(use_tracking)])),
        )

        detect_3d_node_cmd = Node(
            package="fbot_recognition",
            executable="detect_3d_node",
            name="detect_3d_node",
            namespace=namespace,
            additional_env={"PYTHONPATH": new_pythonpath},
            parameters=[
                {
                    "target_frame": target_frame,
                    "depth_image_units_divisor": depth_image_units_divisor,
                    "depth_image_reliability": depth_image_reliability,
                    "depth_info_reliability": depth_info_reliability,
                    "enable_foreground_filter": enable_foreground_filter,
                    "enable_bimodal_filter": enable_bimodal_filter,
                    "mask_erosion_ratio": mask_erosion_ratio,
                    "bimodal_min_gap": bimodal_min_gap,
                    "max_depth_extent_default": max_depth_extent_default,
                    "max_depth_extent_bottle": max_depth_extent_bottle,
                }
            ],
            remappings=[
                ("depth_image", input_depth_topic),
                ("depth_info", input_depth_info_topic),
                ("detections", detect_3d_detections_topic),
                ("detections_3d", detections_3d_topic_value),
            ],
            condition=IfCondition(PythonExpression([str(use_3d)])),
        )

        debug_node_cmd = Node(
            package="fbot_recognition",
            executable="debug_node",
            name="debug_node",
            namespace=namespace,
            additional_env={"PYTHONPATH": new_pythonpath},
            parameters=[
                {
                    "image_reliability": image_reliability,
                    "use_3d": use_3d,
                }
            ],
            remappings=[
                ("image_raw", input_image_topic),
                ("detections", debug_detections_topic),
                ("dbg_image", debug_topic_value),
            ],
            condition=IfCondition(PythonExpression([use_debug])),
        )

        return (
            model_cmd,
            tracker_cmd,
            device_cmd,
            fuse_model_cmd,
            yolo_encoding_cmd,
            enable_cmd,
            threshold_cmd,
            iou_cmd,
            imgsz_height_cmd,
            imgsz_width_cmd,
            half_cmd,
            max_det_cmd,
            augment_cmd,
            agnostic_nms_cmd,
            retina_masks_cmd,
            input_image_topic_cmd,
            image_reliability_cmd,
            input_depth_topic_cmd,
            depth_image_reliability_cmd,
            input_depth_info_topic_cmd,
            depth_info_reliability_cmd,
            target_frame_cmd,
            depth_image_units_divisor_cmd,
            enable_foreground_filter_cmd,
            enable_bimodal_filter_cmd,
            mask_erosion_ratio_cmd,
            bimodal_min_gap_cmd,
            max_depth_extent_default_cmd,
            max_depth_extent_bottle_cmd,
            namespace_cmd,
            use_debug_cmd,
            detections_topic_cmd,
            detections_3d_topic_cmd,
            debug_topic_cmd,
            yolo_node_cmd,
            tracking_node_cmd,
            detect_3d_node_cmd,
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

    return LaunchDescription(
        [
            detections_topic_cmd,
            detections_3d_topic_cmd,
            debug_topic_cmd,
            use_tracking_cmd,
            use_3d_cmd,
            OpaqueFunction(function=run_yolo, args=[use_tracking, use_3d]),
        ]
    )
