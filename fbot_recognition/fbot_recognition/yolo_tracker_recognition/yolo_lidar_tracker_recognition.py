#!/usr/bin/env python

import rclpy
import torch
import cv_bridge
import cv2 as cv
from time import perf_counter
from copy import deepcopy
import numpy as np

# ROS Imports
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from fbot_vision_msgs.msg import Detection2D, Detection2DArray, Detection3D, Detection3DArray, KeyPoint2D, KeyPoint3D
from vision_msgs.msg import BoundingBox3D
from std_msgs.msg import Header
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
import message_filters
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from tf_transformations import euler_from_quaternion, quaternion_from_euler, translation_matrix, quaternion_matrix, inverse_matrix
import ros2_numpy

# YOLO and Custom Recognition Libs
from ultralytics import YOLO
from ReIDManager import ReIDManager
from image2world.image2worldlib import BoundingBoxProcessingData, boundingBoxProcessing, poseProcessing
from fbot_recognition import BaseRecognition
from ament_index_python.packages import get_package_share_directory

class YoloLidarTrackerRecognition(BaseRecognition):
    """
    A ROS2 node for 3D object tracking that fuses 2D detections from YOLO
    with 3D data from a LiDAR sensor and a depth camera.
    """
    def __init__(self, node_name="yolo_lidar_tracker_recognition"):
        super().__init__(nodeName=node_name)
        self.tracking = False
        self.reid_manager = None
        self.lastTrack = perf_counter()
        self.cv_bridge = cv_bridge.CvBridge()

        # TF2 for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.declareParameters()
        self.readParameters()
        self.loadModel()
        self.initRosComm()

        if self.tracking_on_init:
            self.startTracking(Empty.Request(), Empty.Response())
            
        self.get_logger().info(f"Node '{node_name}' started successfully. 🚀")

    def initRosComm(self):
        """Initializes ROS publishers, services, and synchronized subscribers."""
        # Publishers
        self.debugPub = self.create_publisher(Image, self.debug_topic, 1)
        self.recognitionPub = self.create_publisher(Detection2DArray, self.recognition_topic, self.recognition_qs)
        self.recognition3DPub = self.create_publisher(Detection3DArray, self.recognition3D_topic, self.recognition3D_qs)
        self.trackingPub = self.create_publisher(Detection2DArray, self.tracking_topic, self.tracking_qs)
        self.tracking3DPub = self.create_publisher(Detection3DArray, self.tracking3D_topic, self.tracking3D_qs)
        self.markerPublisher = self.create_publisher(MarkerArray, self.markers_topic, self.markers_qs)
        
        # Services
        self.trackingStartService = self.create_service(Empty, self.start_tracking_topic, self.startTracking)
        self.trackingStopService = self.create_service(Empty, self.stop_tracking_topic, self.stopTracking)

        # Synchronized Subscribers for Camera, Depth, and LiDAR
        depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        image_sub = message_filters.Subscriber(self, Image, self.image_topic)
        cam_info_sub = message_filters.Subscriber(self, CameraInfo, self.cam_info_topic)
        lidar_sub = message_filters.Subscriber(self, PointCloud2, self.lidar_topic)

        # Use ApproximateTimeSynchronizer for robustness against minor timestamp differences
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, image_sub, cam_info_sub, lidar_sub],
            queue_size=10,
            slop=0.1  # Allow 0.1s difference
        )
        self.time_synchronizer.registerCallback(self.callback)

    def loadModel(self):
        self.get_logger().info(f"Loading YOLO model: {self.model_file}")
        self.model = YOLO(self.model_file)
        if self.tracking:
            self.loadTrackerModel()

    def loadTrackerModel(self):
        if self.reid_manager is not None:
            return
        self.get_logger().info("Loading Re-ID model for tracking...")
        self.reid_manager = ReIDManager(
            self.reid_model_file, self.reid_model_name, self.reid_threshold,
            self.reid_add_feature_threshold, self.reid_img_size,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.get_logger().info("Re-ID model loaded.")

    def startTracking(self, req: Empty.Request, resp: Empty.Response):
        self.loadTrackerModel()
        self.trackID = -1
        self.lastTrack = perf_counter()
        self.tracking = True
        self.get_logger().info("Tracking service started. 🎯")
        return resp
    
    def stopTracking(self, req: Empty.Request, resp: Empty.Response):
        self.tracking = False
        if self.reid_manager is not None:
            self.reid_manager.clean()
            self.reid_manager = None
        self.get_logger().info("Tracking service stopped. 🛑")
        return resp

    def callback(self, depthMsg: Image, imageMsg: Image, cameraInfoMsg: CameraInfo, lidarMsg: PointCloud2):
        """
        Main processing callback. Receives synchronized sensor data, performs detection,
        fuses LiDAR data, and publishes results.
        """
        HEADER = imageMsg.header
        img_cv = self.cv_bridge.imgmsg_to_cv2(imageMsg)
        debug_img = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)

        # --- YOLO Detection ---
        if self.tracking:
            results = self.model.track(img_cv, persist=True, conf=self.det_threshold, iou=self.iou_threshold,
                                       tracker=self.tracker_cfg_file, verbose=False)
        else:
            results = self.model.predict(img_cv, verbose=False, conf=self.det_threshold)
        
        yolo_result = results[0]
        if yolo_result.boxes is None:
            return

        # --- LiDAR Data Pre-processing ---
        try:
            transform = self.tf_buffer.lookup_transform(
                cameraInfoMsg.header.frame_id,  # Target frame
                lidarMsg.header.frame_id,       # Source frame
                rclpy.time.Time()
            )
            pc_np = ros2_numpy.point_cloud2.pointcloud2_to_xyz_array(lidarMsg)
            pc_camera_frame = self.transform_points(pc_np, transform)
            
            # Camera projection matrix
            K = np.array(cameraInfoMsg.k).reshape(3, 3)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Could not transform point cloud: {e}")
            return

        # --- Process Detections and Fuse with LiDAR ---
        detections2d = []
        detections3d = []
        
        boxes = yolo_result.boxes.data.cpu().numpy()
        ids = yolo_result.boxes.id.cpu().numpy() if self.tracking and yolo_result.boxes.is_track else [-1] * len(boxes)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            score = box[-2]
            class_num = int(box[-1])
            track_id = int(ids[i]) if self.tracking else -1

            # Create 2D Detection Message
            det2d = self.create_detection_2d(box, score, class_num, track_id, HEADER)
            detections2d.append(det2d)
            
            # Fuse with LiDAR to create 3D Bounding Box
            bbox3d_from_lidar, associated_points = self.create_3d_bbox_from_lidar(
                pc_camera_frame, (x1, y1, x2, y2), K
            )

            det3d = None
            if bbox3d_from_lidar:
                # Success: Use LiDAR data
                det3d = self.create_detection_3d(bbox3d_from_lidar, score, HEADER, det2d.label, i, track_id)
            else:
                # Fallback: Use Depth Camera data
                data = BoundingBoxProcessingData()
                data.sensor.setSensorData(cameraInfoMsg, depthMsg)
                data.boundingBox2D = det2d.bbox
                data.maxSize.x, data.maxSize.y, data.maxSize.z = self.max_sizes
                try:
                    bbox3d_from_depth = boundingBoxProcessing(data)
                    # Check if the depth data is valid
                    if np.linalg.norm([bbox3d_from_depth.center.position.x, bbox3d_from_depth.center.position.y, bbox3d_from_depth.center.position.z]) > 0.1:
                        det3d = self.create_detection_3d(bbox3d_from_depth, score, HEADER, det2d.label, i, track_id)
                except Exception as e:
                    self.get_logger().warn(f"Depth processing failed: {e}")

            if det3d:
                detections3d.append(det3d)

            # --- Visualization ---
            box_label = f"ID:{track_id} " if self.tracking else ""
            cv.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.putText(debug_img, f"{box_label}{self.model.names[class_num]}:{score:.2f}",
                       (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # --- Publish Results ---
        self.publish_all(detections2d, detections3d, imageMsg, debug_img, HEADER)

    def create_detection_2d(self, box, score, class_num, track_id, header):
        """Helper to create a Detection2D message."""
        x1, y1, x2, y2 = box[:4]
        det = Detection2D()
        det.header = header
        det.bbox.center.position.x = float(x1 + x2) / 2.0
        det.bbox.center.position.y = float(y1 + y2) / 2.0
        det.bbox.size_x = float(x2 - x1)
        det.bbox.size_y = float(y2 - y1)
        det.label = self.model.names[class_num]
        det.score = float(score)
        det.class_num = class_num
        det.global_id = track_id
        return det

    def create_3d_bbox_from_lidar(self, points_camera_frame, bbox2d, k_matrix):
        """
        Projects LiDAR points into the image, finds points within the 2D bbox,
        and computes a 3D bounding box from them.
        """
        x1, y1, x2, y2 = bbox2d
        
        # Filter points in front of the camera
        points_in_front = points_camera_frame[points_camera_frame[:, 2] > 0]
        if len(points_in_front) == 0:
            return None, None

        # Project points to 2D image plane
        projected_points = (k_matrix @ points_in_front.T).T
        u = projected_points[:, 0] / projected_points[:, 2]
        v = projected_points[:, 1] / projected_points[:, 2]
        
        # Find points inside the 2D bounding box
        mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        associated_points = points_in_front[mask]

        if associated_points.shape[0] < 10:  # Require a minimum number of points
            return None, None
            
        # Create 3D bounding box from the associated points
        min_coords = np.min(associated_points, axis=0)
        max_coords = np.max(associated_points, axis=0)
        
        bbox3d = BoundingBox3D()
        bbox3d.center.position.x = float((min_coords[0] + max_coords[0]) / 2.0)
        bbox3d.center.position.y = float((min_coords[1] + max_coords[1]) / 2.0)
        bbox3d.center.position.z = float((min_coords[2] + max_coords[2]) / 2.0)
        
        bbox3d.size.x = float(max_coords[0] - min_coords[0])
        bbox3d.size.y = float(max_coords[1] - min_coords[1])
        bbox3d.size.z = float(max_coords[2] - min_coords[2])

        return bbox3d, associated_points

    def create_detection_3d(self, bb3d: BoundingBox3D, score: float, header: Header, label: str, id: int, global_id: int):
        """Helper to create a Detection3D message."""
        detection3d = Detection3D()
        detection3d.header = header
        detection3d.id = id
        detection3d.global_id = global_id
        detection3d.label = label
        detection3d.score = score
        detection3d.bbox3d = bb3d
        return detection3d

    def publish_all(self, detections2d, detections3d, image_msg, debug_img, header):
        """Publishes all detection and debug messages."""
        # 2D Detections
        det2d_array = Detection2DArray(header=header, detections=detections2d, image_rgb=image_msg)
        self.recognitionPub.publish(det2d_array)

        # 3D Detections
        det3d_array = Detection3DArray(header=header, detections=detections3d, image_rgb=image_msg)
        if detections3d:
            self.recognition3DPub.publish(det3d_array)
            self.publishMarkers(detections3d)

        # Publish tracking-specific messages if tracking is active
        if self.tracking:
            # Here you can add logic to identify the primary tracked person
            # For simplicity, we'll publish all tracked items
            tracked_2d = [d for d in detections2d if d.global_id != -1]
            tracked_3d = [d for d in detections3d if d.global_id != -1]
            if tracked_2d:
                self.trackingPub.publish(Detection2DArray(header=header, detections=tracked_2d, image_rgb=image_msg))
            if tracked_3d:
                self.tracking3DPub.publish(Detection3DArray(header=header, detections=tracked_3d, image_rgb=image_msg))

        # Debug Image
        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_img, "bgr8")
        debug_msg.header = header
        self.debugPub.publish(debug_msg)

    def publishMarkers(self, detections3d, color=[0.0, 1.0, 0.0]): # Green for LiDAR-fused
        """Publishes visualization markers for RViz."""
        markers = MarkerArray()
        duration = Duration(sec=1)
        for i, det in enumerate(detections3d):
            marker = Marker()
            marker.header = det.header
            marker.action = Marker.ADD
            marker.pose = det.bbox3d.center
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 0.5
            marker.ns = "fused_bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.scale = det.bbox3d.size
            marker.lifetime = duration
            markers.markers.append(marker)
        self.markerPublisher.publish(markers)
    
    def transform_points(self, points, transform):
        """Applies a TF2 transform to a numpy array of points."""
        t = transform.transform.translation
        q = transform.transform.rotation
        
        trans_matrix = translation_matrix([t.x, t.y, t.z])
        rot_matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
        transform_matrix = np.dot(trans_matrix, rot_matrix)

        # Add a homogeneous coordinate (1) to each point
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Apply transformation
        transformed_points_h = (transform_matrix @ points_h.T).T
        
        # Return to 3D coordinates
        return transformed_points_h[:, :3]

    def declareParameters(self):
        super().declareParameters()
        self.declare_parameter("subscribers.lidar.topic", "/lidar/points")
        
        self.declare_parameter("publishers.debug.topic","/fbot_vision/fr/debug")
        self.declare_parameter("publishers.recognition.topic", "/fbot_vision/fr/recognition2D")
        self.declare_parameter("publishers.recognition.qos_profile", 10)
        self.declare_parameter("publishers.recognition3D.topic", "/fbot_vision/fr/recognition3D")
        self.declare_parameter("publishers.recognition3D.qos_profile", 10)
        self.declare_parameter("publishers.tracking.topic", "/fbot_vision/pt/tracking2D")
        self.declare_parameter("publishers.tracking.qos_profile", 10)
        self.declare_parameter("publishers.tracking3D.topic", "/fbot_vision/pt/tracking3D")
        self.declare_parameter("publishers.tracking3D.qos_profile", 10)
        self.declare_parameter("publishers.markers.topic", "/fbot_vision/fr/markers")
        self.declare_parameter("publishers.markers.qos_profile", 10)

        self.declare_parameter("services.tracking.start","/fbot_vision/pt/start")
        self.declare_parameter("services.tracking.stop","/fbot_vision/pt/stop")

        self.declare_parameter("model_file","yolov8n-pose.pt")
        self.declare_parameter("tracking.reid.model_file","osnet_x0_25_msmt17.pt")
        self.declare_parameter("tracking.reid.model_name","osnet_x0_25")
        
        self.declare_parameter("tracking.thresholds.detection", 0.5)
        self.declare_parameter("tracking.thresholds.reid", 0.75)
        self.declare_parameter("tracking.thresholds.reid_feature_add",0.7)
        self.declare_parameter('tracking.thresholds.iou',0.5)
        self.declare_parameter("tracking.start_on_init", False)

        self.declare_parameter("tracking.reid.img_size.height",256)
        self.declare_parameter("tracking.reid.img_size.width",128)
        self.declare_parameter("tracking.config_file","botsort.yaml")
        self.declare_parameter("max_sizes", [2.5, 2.5, 2.5])

    def readParameters(self):
        super().readParameters()
        self.lidar_topic = self.get_parameter("subscribers.lidar.topic").value

        self.debug_topic = self.get_parameter("publishers.debug.topic").value
        self.recognition_topic = self.get_parameter("publishers.recognition.topic").value
        self.recognition_qs = self.get_parameter("publishers.recognition.qos_profile").value
        self.recognition3D_topic = self.get_parameter("publishers.recognition3D.topic").value
        self.recognition3D_qs = self.get_parameter("publishers.recognition3D.qos_profile").value
        self.start_tracking_topic = self.get_parameter("services.tracking.start").value
        self.stop_tracking_topic = self.get_parameter("services.tracking.stop").value
        self.tracking_topic = self.get_parameter("publishers.tracking.topic").value
        self.tracking_qs = self.get_parameter("publishers.tracking.qos_profile").value
        self.tracking3D_topic = self.get_parameter("publishers.tracking3D.topic").value
        self.tracking3D_qs = self.get_parameter("publishers.tracking3D.qos_profile").value
        self.markers_topic = self.get_parameter("publishers.markers.topic").value
        self.markers_qs = self.get_parameter("publishers.markers.qos_profile").value

        share_dir = get_package_share_directory("fbot_recognition")
        self.model_file = f"{share_dir}/weights/{self.get_parameter('model_file').value}"
        self.reid_model_file = f"{share_dir}/weights/{self.get_parameter('tracking.reid.model_file').value}"
        self.reid_model_name = self.get_parameter("tracking.reid.model_name").value
        self.tracker_cfg_file = f"{share_dir}/config/yolo_tracker_config/{self.get_parameter('tracking.config_file').value}"
        
        self.det_threshold = self.get_parameter("tracking.thresholds.detection").value
        self.reid_threshold = self.get_parameter("tracking.thresholds.reid").value
        self.reid_add_feature_threshold = self.get_parameter("tracking.thresholds.reid_feature_add").value
        self.iou_threshold = self.get_parameter('tracking.thresholds.iou').value
        self.tracking_on_init = self.get_parameter("tracking.start_on_init").value
        self.reid_img_size = (self.get_parameter("tracking.reid.img_size.height").value, self.get_parameter("tracking.reid.img_size.width").value)
        self.max_sizes = self.get_parameter("max_sizes").value

def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloLidarTrackerRecognition()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()