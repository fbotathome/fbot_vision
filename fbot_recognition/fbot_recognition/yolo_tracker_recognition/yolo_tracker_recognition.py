#!/usr/bin/env python

import rclpy
import torch
import cv_bridge
import cv2 as cv
from time import perf_counter

from copy import deepcopy

from ultralytics import YOLO
from ReIDManager import ReIDManager
from image2world import BoundingBoxProcessingData, boundingBoxProcessing, poseProcessing
from fbot_recognition import BaseRecognition
import numpy as np

from fbot_vision_msgs.msg import Detection2D, Detection2DArray, Detection3D, Detection3DArray, KeyPoint2D, KeyPoint3D
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from vision_msgs.msg import BoundingBox2D, BoundingBox3D
from std_msgs.msg import Header
from std_srvs.srv import Empty

from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory
import message_filters
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import sensor_msgs_py.point_cloud2 as pc2

# Add TF2 imports
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

SOURCES_TYPES = {
        'camera_info': CameraInfo,
        'image_rgb': Image,
        'image_depth': Image
    }

class YoloTrackerRecognition(BaseRecognition):
    def __init__(self, node_name):
        super().__init__(nodeName=node_name)
        # Initialize TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.tracking = False
        self.reid_manager = None
        self.lastTrack = perf_counter()
        self.cv_bridge = cv_bridge.CvBridge()
        self.declareParameters()
        self.readParameters()
        self.loadModel()
        self.initRosComm()
        if self.tracking_on_init:
            self.startTracking()
        self.get_logger().info(f"Node started!!!")
        # self.loadModel()

    def initRosComm(self):
        self.syncSubscribers()
        self.debugPub = self.create_publisher(Image, self.debug_topic, qos_profile=1)
        self.recognitionPub = self.create_publisher(Detection2DArray, self.recognition_topic, qos_profile=self.recognition_qs)
        self.recognition3DPub = self.create_publisher(Detection3DArray, self.recognition3D_topic, qos_profile=self.recognition3D_qs)
        self.trackingPub = self.create_publisher(Detection2DArray, self.tracking_topic, qos_profile=self.tracking_qs)
        self.tracking3DPub = self.create_publisher(Detection3DArray, self.tracking3D_topic, qos_profile=self.tracking3D_qs)
        self.markerPublisher = self.create_publisher(MarkerArray,self.markers_topic,qos_profile=self.markers_qs)
        # Optional lidar debug publisher
        if self.debug_publish_lidar:
            self.lidarDebugPub = self.create_publisher(PointCloud2, self.lidar_debug_topic, qos_profile=self.lidar_debug_qs)
        else:
            self.lidarDebugPub = None
        self.trackingStartService = self.create_service(Empty, self.start_tracking_topic, self.startTracking)
        self.trackingStopService = self.create_service(Empty, self.stop_tracking_topic, self.stopTracking)
    
    def syncSubscribers(self):
        subscribers = []
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=self.qosProfile,reliability=QoSReliabilityPolicy.BEST_EFFORT)
        
        # Ensure a consistent order for subscribers
        sub_topics = ['image_rgb', 'camera_info', 'image_depth']
        
        for topic in sub_topics:
            if topic in self.topicsToSubscribe and topic in SOURCES_TYPES:
                subscribers.append(message_filters.Subscriber(self, SOURCES_TYPES[topic], self.topicsToSubscribe[topic], qos_profile=qos))

        # Add lidar subscriber
        subscribers.append(message_filters.Subscriber(self, PointCloud2, self.lidar_points_topic, qos_profile=qos))
        
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=1, slop=self.slop)
        self._synchronizer.registerCallback(self.callback)
    
    def loadModel(self):
        self.get_logger().info(f"Loading model: {self.model_file}")
        self.model = YOLO(self.model_file)
        if self.tracking:
            self.loadTrackerModel()

    def loadTrackerModel(self):
        if self.reid_manager != None:
            return
        self.reid_manager = ReIDManager(
            self.reid_model_file,
            self.reid_model_name,
            self.reid_threshold,
            self.reid_add_feature_threshold,
            self.reid_img_size
        )

    def unLoadModel(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = None
        self.loadTrackerModel()
        return
    
    def unLoadTrackerModel(self):
        if self.reid_manager == None:
            return
        self.reid_manager.clean()
    
    def startTracking(self,req : Empty.Request, resp : Empty.Response):
        self.loadTrackerModel()
        self.trackID = -1
        self.lastTrack = perf_counter()
        self.tracking = True
        self.get_logger().info("Tracking started!!!")
        return resp
    
    def stopTracking(self, req : Empty.Request, resp : Empty.Response):
        self.tracking = False
        self.unLoadTrackerModel()
        self.get_logger().info("Tracking stoped!!!")
        return resp

    def extractLidarDepth(self, pointCloudMsg: PointCloud2, bbox: BoundingBox2D, cameraInfo: CameraInfo, transform=None):
        """
        Extract median depth from lidar point cloud for a given 2D bounding box.
        Expects point cloud to already be in camera optical frame.
        Returns median depth in meters, the projected 2D points, and the filtered 3D points.
        """
        # Convert point cloud to numpy array once
        pc_data = pc2.read_points(pointCloudMsg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(pc_data))
        
        if len(points) == 0:
            return None, None, None
        
        # If transform provided, apply it to all points at once using matrix operations
        if transform is not None:
            # Convert quaternion to rotation matrix
            q = transform.transform.rotation
            rotation_matrix = np.array([
                [1 - 2*q.y*q.y - 2*q.z*q.z, 2*q.x*q.y - 2*q.z*q.w, 2*q.x*q.z + 2*q.y*q.w],
                [2*q.x*q.y + 2*q.z*q.w, 1 - 2*q.x*q.x - 2*q.z*q.z, 2*q.y*q.z - 2*q.x*q.w],
                [2*q.x*q.z - 2*q.y*q.w, 2*q.y*q.z + 2*q.x*q.w, 1 - 2*q.x*q.x - 2*q.y*q.y]
            ])
            
            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            # Apply transformation: rotation @ points.T + translation
            points_transformed = rotation_matrix @ points.T + translation[:, np.newaxis]
            points = points_transformed.T
        
        # Pre-compute camera intrinsics for efficiency
        fx = cameraInfo.k[0]
        fy = cameraInfo.k[4]
        cx = cameraInfo.k[2]
        cy = cameraInfo.k[5]
        
        # Vectorized projection for all points at once
        z_valid = points[:, 2] > 0.1  # min depth 10cm
        if not np.any(z_valid):
            return None, None, None
            
        valid_points = points[z_valid]
        u_all = (valid_points[:, 0] * fx / valid_points[:, 2]) + cx
        v_all = (valid_points[:, 1] * fy / valid_points[:, 2]) + cy
        
        # Bounding box bounds
        x_min = bbox.center.position.x - bbox.size_x / 2
        x_max = bbox.center.position.x + bbox.size_x / 2
        y_min = bbox.center.position.y - bbox.size_y / 2
        y_max = bbox.center.position.y + bbox.size_y / 2
        
        # Filter points within bounding box
        bbox_mask = (u_all >= x_min) & (u_all <= x_max) & (v_all >= y_min) & (v_all <= y_max)
        
        if not np.any(bbox_mask):
            return None, None, None
            
        depths = valid_points[bbox_mask, 2]
        points_3d_in_bbox = valid_points[bbox_mask]
        
        # Use more robust depth estimation with outlier filtering
        if len(depths) > 3:
            q75, q25 = np.percentile(depths, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            filtered_depths = depths[(depths >= lower_bound) & (depths <= upper_bound)]
            if len(filtered_depths) > 0:
                median_depth = np.median(filtered_depths)
            else:
                median_depth = np.median(depths)  # fallback
        else:
            median_depth = np.median(depths)
        
        used_points_2d = np.column_stack((u_all[bbox_mask], v_all[bbox_mask]))
        return median_depth, used_points_2d, points_3d_in_bbox

    def callback(self, imageMsg: Image, cameraInfoMsg: CameraInfo, depthMsg: Image, pointCloudMsg: PointCloud2):
        tracking = self.tracking

        img = imageMsg
        img_depth = depthMsg
        camera_info = cameraInfoMsg
        HEADER = img.header

        # Cache TF transform for this callback
        cached_transform = None
        
        try:
            # Get transform once per callback if point cloud is in a different frame
            target_frame = self.camera_optical_frame
            source_frame = pointCloudMsg.header.frame_id
            if source_frame != target_frame:
                cached_transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    pointCloudMsg.header.stamp,
                    rclpy.duration.Duration(seconds=1.0)
                )
                self.get_logger().debug(f"Cached transform from {source_frame} to {target_frame}")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Could not get transform from {pointCloudMsg.header.frame_id} to {self.camera_optical_frame}: {e}")
            # Continue without lidar data, will fallback to camera depth

        recognition = Detection2DArray()
        recognition.image_rgb = img

        recognition3D = Detection3DArray()
        recognition3D.image_rgb = img
        data = BoundingBoxProcessingData()
        
        original_depth_msg = img_depth
        data.sensor.setSensorData(camera_info, original_depth_msg)

        recognition.header = HEADER
        recognition.detections = []
        img = self.cv_bridge.imgmsg_to_cv2(img)
        debug_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert once at the beginning
        results = None
        bboxs   = None

        if tracking:
            results = list(self.model.track(img, persist=True,
                                        conf=self.det_threshold,
                                        iou=self.iou_threshold,
                                        device="cuda:0",
                                        tracker=self.tracker_cfg_file,
                                        verbose=True, stream=True))
            bboxs = results[0].boxes.data.cpu().numpy()
        else:
            results = list(self.model.predict(img, verbose=False, stream=True))
            bboxs = results[0].boxes.data.cpu().numpy()

        # Early return if no detections
        if len(bboxs) == 0:
            return

        people_ids = []

        tracked_box = None
        now = perf_counter()
        is_aged = (now - self.lastTrack >= self.max_time)
        is_id_found = False
        previus_size = float("-inf")
        new_id = -1
        # descriptions = []
        ids = []
        if results[0].boxes.is_track:
            img_patchs = []
            for x1,y1,x2,y2 in results[0].boxes.xyxy.cpu().numpy():
                img_patchs.append(img[int(y1):int(y2),int(x1):int(x2)])
            ids = self.reid_manager.extract_ids(results[0].boxes.id.cpu().numpy(),img_patchs)
        for i, box in enumerate(bboxs):
            description = Detection2D()
            description.header = HEADER

            X1,Y1,X2,Y2 = box[:4]
            ID = int(ids[i]) if self.tracking and len(box) == 7 else -1
            score = box[-2]
            clss = int(box[-1])

            description.bbox.center.position.x = float(X1+X2)/2.0
            description.bbox.center.position.y = float(Y1+Y2)/2.0
            description.bbox.size_x = float(X2-X1)
            description.bbox.size_y = float(Y2-Y1)
            description.label = self.model.names[clss]
            description.type = Detection2D.DETECTION
            description.score = float(score)
            description.class_num = clss
            description.id = i

            box_label = ""
            if tracking:
                description.global_id = ID
                if description.label == "person":
                    people_ids.append(ID)                 
                
                box_label = f"ID:{ID} "
                size = description.bbox.size_x * description.bbox.size_y

                if ID == self.trackID:
                    is_id_found = True
                    tracked_box = description

                if (not is_id_found) and (is_aged or self.trackID == -1):
                    if tracked_box is None or size > previus_size:
                        previus_size = size
                        tracked_box = description
                        new_id = ID          

            recognition.detections.append(description)
            
            

            cv.rectangle(debug_img,(int(X1),int(Y1)), (int(X2),int(Y2)),(0,0,255),thickness=2)
            cv.putText(debug_img,f"{box_label}{self.model.names[clss]}:{score:.2f}", (int(X1), int(Y1)), cv.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),thickness=2)
        
        track_recognition = Detection2DArray()
        track_recognition.header = HEADER
        tracked_description : Detection2D = deepcopy(tracked_box)
        if tracked_box is not None:
            track_recognition.header = recognition.header
            track_recognition.image_rgb = recognition.image_rgb
            tracked_description.type = Detection2D.DETECTION
            if not is_id_found:
                self.trackID = new_id
            self.lastTrack = now
            cv.rectangle(debug_img,(int(tracked_box.bbox.center.position.x-tracked_box.bbox.size_x/2),\
                                    int(tracked_box.bbox.center.position.y-tracked_box.bbox.size_y/2)),\
                                    (int(tracked_box.bbox.center.position.x+tracked_box.bbox.size_x/2),\
                                    int(tracked_box.bbox.center.position.y+tracked_box.bbox.size_y/2)),(255,0,0),thickness=2)
            
        
        if results[0].keypoints != None:
            poses = results[0].keypoints.data.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            poses_idx = scores > self.det_threshold
            poses = poses[poses_idx]
            counter = 0
            # if not tracking or len(people_ids) == len(poses):
            desc : Detection2D
            for desc in recognition.detections:
                if desc.label == "person" and desc.score >= self.det_threshold:
                    desc.type = Detection2D.POSE
                    # rospy.logwarn(desc.score)
                    for idx, kpt in enumerate(poses[counter]):
                        keypoint = KeyPoint2D()
                        keypoint.x = float(kpt[0])
                        keypoint.y = float(kpt[1])
                        keypoint.id = idx
                        keypoint.score = float(kpt[2])
                        desc.pose.append(keypoint)
                        if kpt[2] >= self.threshold:
                            cv.circle(debug_img, (int(kpt[0]), int(kpt[1])),3,(0,255,0),thickness=-1)
                        if tracking:
                            desc.global_id = people_ids[counter]
                    if tracked_box != None and tracked_description.global_id == desc.global_id:
                        desc.header = HEADER
                        tracked_description = desc
                        for kpt in desc.pose:
                            if kpt.score >= self.threshold:
                                cv.circle(debug_img, (int(kpt.x), int(kpt.y)),3,(0,255,255),thickness=-1)
                    counter +=1
        
        description : Detection2D
        track_recognition3D = Detection3DArray()
        track_recognition3D.image_rgb = recognition.image_rgb
        track_recognition3D.header = HEADER
        for description in recognition.detections:
            #3D
            data.boundingBox2D.center.position.x = description.bbox.center.position.x 
            data.boundingBox2D.center.position.y = description.bbox.center.position.y
            data.boundingBox2D.size_x = description.bbox.size_x
            data.boundingBox2D.size_y = description.bbox.size_y
            data.maxSize.x, data.maxSize.y, data.maxSize.z = self.max_sizes
            
            # Try to get lidar depth
            lidar_depth, lidar_points_2d, lidar_points_3d = self.extractLidarDepth(pointCloudMsg, data.boundingBox2D, camera_info, cached_transform)
            
            if lidar_depth is not None and lidar_points_3d is not None:
                # If lidar provides depth, create a temporary depth image with this data
                depth_cv = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
                x1 = int(description.bbox.center.position.x - description.bbox.size_x / 2)
                x2 = int(description.bbox.center.position.x + description.bbox.size_x / 2)
                y1 = int(description.bbox.center.position.y - description.bbox.size_y / 2)
                y2 = int(description.bbox.center.position.y + description.bbox.size_y / 2)
                # Clip to image bounds
                x1, x2 = max(0, x1), min(depth_cv.shape[1], x2)
                y1, y2 = max(0, y1), min(depth_cv.shape[0], y2)

                if x2 > x1 and y2 > y1:
                    depth_cv[y1:y2, x1:x2] = int(lidar_depth * 1000)  # Convert meters to millimeters
                    modified_depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_cv, encoding="passthrough")
                    modified_depth_msg.header = HEADER
                    data.sensor.setSensorData(camera_info, modified_depth_msg)
                    
                    # Publish the 3D points used for debugging
                    if self.debug_publish_lidar and self.lidarDebugPub is not None:
                        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.camera_optical_frame)
                        pc_msg = pc2.create_cloud_xyz32(header, lidar_points_3d)
                        self.lidarDebugPub.publish(pc_msg)

                        # Visual feedback for debugging: draw 2D projected lidar points
                        if lidar_points_2d is not None:
                            for point in lidar_points_2d:
                                px, py = int(point[0]), int(point[1])
                                if 0 <= px < debug_img.shape[1] and 0 <= py < debug_img.shape[0]:
                                    cv.circle(debug_img, (px, py), 2, (255, 0, 0), -1)
                    self.get_logger().debug(f"Used lidar depth {lidar_depth:.2f}m for detection.")
            else:
                # Fallback to original camera depth if lidar fails
                data.sensor.setSensorData(camera_info, original_depth_msg)
                self.get_logger().debug(f"Lidar depth failed, using camera depth as fallback.")

            try:
                bbox3D = boundingBoxProcessing(data)
            except Exception as e:
                self.get_logger().warn(str(e))
                continue
            pose3D = []

            if results[0].keypoints != None:
                data.pose = [(kp.x, kp.y, kp.score, kp.id) for kp in description.pose]
                pose3D = poseProcessing(data)
            description3D = self.createDetection3d(bbox3D,description.score,HEADER,description.label, description.id, description.global_id, pose=pose3D)
            if tracked_description != None and description.global_id == tracked_description.global_id:
                track_recognition3D.detections.append(description3D)
            
            recognition3D.detections.append(description3D)

        if tracked_description is not None:
            track_recognition.detections.append(tracked_description)
        
        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_img, "bgr8")
        debug_msg.header = HEADER
        self.debugPub.publish(debug_msg)
        
        if len(recognition.detections) > 0:
            self.recognitionPub.publish(recognition)
        if len(recognition3D.detections) > 0:
            self.recognition3DPub.publish(recognition3D)
            self.publishMarkers(recognition3D.detections)
        if tracked_box != None and len(track_recognition.detections) > 0:
            self.trackingPub.publish(track_recognition)
            self.tracking3DPub.publish(track_recognition3D)

    def createDetection3d(self, bb3d: BoundingBox3D , score: float, detectionHeader: Header, label: str, id : int = 0 , global_id : int=0, pose : list = []) -> Detection3D:
        detection3d = Detection3D()
        detection3d.header = detectionHeader
        # Ensure frame_id is set for proper TF transformations
        if not detection3d.header.frame_id:
            detection3d.header.frame_id = self.camera_optical_frame
        # Set poses_header for compatibility with TrackedPersonPoseState
        detection3d.poses_header = detectionHeader
        detection3d.id = id
        detection3d.global_id = global_id
        detection3d.label = label
        detection3d.score = score
        detection3d.bbox3d = bb3d

        if len(pose) != 0:
            for kpt in pose:
                # print(kpt)
                kpt3D = KeyPoint3D()
                kpt3D.x = kpt[0]
                kpt3D.y = kpt[1]
                kpt3D.z = kpt[2]
                kpt3D.score = kpt[3]
                kpt3D.id = kpt[4]
                detection3d.pose.append(kpt3D)

        return detection3d
    
    def publishMarkers(self, descriptions3d : Detection3DArray , color=[255,0,0]):
        markers = MarkerArray()
        duration = Duration()
        duration.sec = 2
        color = np.asarray(color)/255.0
        for i, det in enumerate(descriptions3d):
            # det.header.frame_id = "map"
            name = det.label

            # cube marker
            marker = Marker()
            marker.header = det.header
            marker.action = Marker.ADD
            marker.pose = det.bbox3d.center
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.4
            marker.ns = "bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.scale = det.bbox3d.size
            marker.lifetime = duration
            markers.markers.append(marker)

            # text marker
            marker = Marker()
            marker.header = det.header
            marker.action = Marker.ADD
            marker.pose = det.bbox3d.center
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker.id = i
            marker.ns = "texts"
            marker.type = Marker.TEXT_VIEW_FACING
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.text = '{} ({:.2f})'.format(name, det.score)
            marker.lifetime = duration
            markers.markers.append(marker)

            for idx, kpt3D in enumerate(det.pose):
                # print("oi")
                if kpt3D.score > 0:
                    marker = Marker()
                    marker.header = det.header
                    marker.type = Marker.SPHERE
                    marker.id = idx
                    marker.color.r = color[1]
                    marker.color.g = color[2]
                    marker.color.b = color[0]
                    marker.color.a = 1.0
                    marker.scale.x = 0.05
                    marker.scale.y = 0.05
                    marker.scale.z = 0.05
                    marker.pose.position.x = kpt3D.x
                    marker.pose.position.y = kpt3D.y
                    marker.pose.position.z = kpt3D.z
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.lifetime = duration
                    markers.markers.append(marker)
        
        self.markerPublisher.publish(markers)
    
    def declareParameters(self):
        super().declareParameters()
        self.declare_parameter("publishers.debug.topic","/fbot_vision/fr/debug")

        # Lidar debug controls
        self.declare_parameter("debug.publish_lidar_points", True)
        self.declare_parameter("publishers.lidar_debug.topic", "/fbot_vision/fr/lidar_points_debug")
        self.declare_parameter("publishers.lidar_debug.qos_profile", 10)

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

        self.declare_parameter("subscribers.lidar_points", "/fbot_vision/lidar/points")

        self.declare_parameter("camera_optical_frame", "camera_color_optical_frame")

        self.declare_parameter("debug_kpt_threshold", 0.5)

        self.declare_parameter("model_file","yolo11n-pose")
        self.declare_parameter("tracking.reid.model_file","resnet_reid_model.pt")
        self.declare_parameter("tracking.reid.model_name","resnet50")

        self.declare_parameter("tracking.thresholds.detection", 0.5)
        self.declare_parameter("tracking.thresholds.reid", 0.75)
        self.declare_parameter("tracking.thresholds.reid_feature_add",0.7)
        self.declare_parameter('tracking.thresholds.iou',0.5)
        self.declare_parameter("tracking.thresholds.max_time",60)
        self.declare_parameter("tracking.thresholds.max_age",5)
        self.declare_parameter("tracking.start_on_init", False)

        self.declare_parameter("tracking.reid.img_size.height",256)
        self.declare_parameter("tracking.reid.img_size.width",128)

        self.declare_parameter("tracking.config_file","yolo_tracker_default_config.yaml")

        self.declare_parameter("max_sizes", [2.5, 2.5, 2.5])

        # self.declare_parameter("tracking.model_name",128)

        return 
    
    def readParameters(self):
        super().readParameters()
        self.debug_topic = self.get_parameter("publishers.debug.topic").value

        # Lidar debug controls
        self.debug_publish_lidar = self.get_parameter("debug.publish_lidar_points").value
        self.lidar_debug_topic = self.get_parameter("publishers.lidar_debug.topic").value
        self.lidar_debug_qs = self.get_parameter("publishers.lidar_debug.qos_profile").value

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

        self.threshold = self.get_parameter("debug_kpt_threshold").value

        self.lidar_points_topic = self.get_parameter("subscribers.lidar_points").value
        self.camera_optical_frame = self.get_parameter("camera_optical_frame").value

        share_directory = get_package_share_directory("fbot_recognition")

        self.model_file = share_directory + "/weights/" + self.get_parameter("model_file").value
        self.reid_model_file = share_directory + "/weights/" + self.get_parameter("tracking.reid.model_file").value
        self.reid_model_name = self.get_parameter("tracking.reid.model_name").value

        self.det_threshold = self.get_parameter("tracking.thresholds.detection").value
        self.reid_threshold = self.get_parameter("tracking.thresholds.reid").value
        self.reid_add_feature_threshold = self.get_parameter("tracking.thresholds.reid_feature_add").value
        self.iou_threshold = self.get_parameter('tracking.thresholds.iou').value
        self.max_time = self.get_parameter("tracking.thresholds.max_time").value
        self.max_age = self.get_parameter("tracking.thresholds.max_age").value
        self.tracking_on_init = self.get_parameter("tracking.start_on_init").value
        self.tracker_cfg_file = share_directory + "/config/yolo_tracker_config/" + self.get_parameter("tracking.config_file").value

        self.reid_img_size = (self.get_parameter("tracking.reid.img_size.height").value,self.get_parameter("tracking.reid.img_size.width").value)

        self.max_sizes = self.get_parameter("max_sizes").value

    

def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloTrackerRecognition("yolo_tracker_recognition")

    try:
        while rclpy.ok():
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()