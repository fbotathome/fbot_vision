#!/usr/bin/env python3
import cv2
import numpy as np
import message_filters
import rclpy
from rclpy.node import Node
from ultralytics import YOLO
from ultralytics.engine.results import Results
from sensor_msgs.msg import Image, CameraInfo  # Changed to Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from image2world import Image2World

class CamRgbLidarDetection(Node):

    def _init_(self):
        super()._init_(node_name='cam_rgb_lidar_detection')  # type: ignore
        self.index = 0
        # subs
        depth_sub = message_filters.Subscriber(
            self, Image, "/camera/aligned_depth_to_color/image_raw", qos_profile=15)
        
        camera_info_sub = message_filters.Subscriber(
            self, CameraInfo, "/camera/color/camera_info", qos_profile=15)
        
        image_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw', qos_profile=15)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (depth_sub, image_sub, camera_info_sub), 15, 2)

        self.get_logger().info("Image 2 world: Sincronizing topics")        
        self._synchronizer.registerCallback(self.callback)
        # self.subscription = self.create_subscription(
        #     Image,  
        #     '/camera/color/image_raw',  
        #     self.callback, 
        #     1
        # )
        self.debug_image = self.create_publisher(Marker, "/cam_rgb_lidar/detections/image", 1)  
        self.debug_yolo = self.create_publisher(Image, "/cam_rgb_lidar/detections/yolo", 1)

        self.logParams()
        self.readParams()
        self.loadModel()
        self.loadCVBrige()

    def logParams(self):
        parameters = self.get_parameters_by_prefix('')
        # for parameter in parameters:
        #     self.get_logger().info(f"Parameter: {parameter}")

    def readParams(self):
        pass
        # self.network_weight_path = '/home/digitaltwin/icrane_vision_ws/src/icrane_vision/weights/seg_box.pt'

    def loadCVBrige(self):
        self.cv_bridge = CvBridge()

    def callback(self, depth_msg: Image, image_msg: Image, camera_info_msg: CameraInfo):
        img = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8') 
        result = self.model(img)
        results: Results = result[0].cpu()

        debug_img = results.plot()
        debug_img = self.cv_bridge.cv2_to_imgmsg(debug_img, encoding="passthrough")
        self.debug_yolo.publish(debug_img)

        if results.boxes is None or len(results.boxes.xywh) == 0:
            return
        
        center_x, center_y, size_x, size_y = map(float, results.boxes.xywh[0])

        i2w = Image2World()

        data = image2worldlib.BoundingBoxProcessingData()

        # center = image2worldlib.Center(x=results[0].boxes.xywh[0], y=results[0].boxes.xywh[1], z=0)
        # size = image2worldlib.Size(width=results[0].boxes.xywh[2], height=results[0].boxes.xywh[3], depth=0)
        # maxSize = image2worldlib.Vector3(x=5.0, y=5.0, z=5.0)
        data.boundingBox2D.center.x = center_x
        data.boundingBox2D.center.y = center_y
        data.boundingBox2D.size_x = size_x
        data.boundingBox2D.size_y = size_y
        data.maxSize.x = 100.
        data.maxSize.y = 100.
        data.maxSize.z = 100.

        data.sensor.setSensorData(imageDepth=depth_msg, cameraInfo=camera_info_msg)

        try:
            bb3d = i2w.boundingBoxProcessing(data)
        except Exception as e:
            self.get_logger().error(f"Error processing bounding box: {e}")
            return

        marker = self.createMarker(bb3d)

        self.debug_image.publish(marker)

    def createMarker(self, bb3d: image2worldlib.BoundingBox3D):
        marker = Marker()
        marker.header.frame_id = "camera_color_optical_frame"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = bb3d.center.position.x
        marker.pose.position.y = bb3d.center.position.y
        marker.pose.position.z = bb3d.center.position.z
        marker.pose.orientation.x = bb3d.center.orientation.x
        marker.pose.orientation.y = bb3d.center.orientation.y
        marker.pose.orientation.z = bb3d.center.orientation.z
        marker.pose.orientation.w = bb3d.center.orientation.w
        marker.scale.x = bb3d.size.x
        marker.scale.y = bb3d.size.y
        marker.scale.z = bb3d.size.z
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        return marker

    def loadModel(self):
        self.get_logger().info("=> Loading model")
        self.model = YOLO('/home/eduardo/icrane_ws/src/icrane_vision/icrane_vision/yolov8n-face.pt')
        self.get_logger().info("=> Loaded")

def main(args=None):
    rclpy.init(args=args)
    cam_rgb_lidar_detection = CamRgbLidarDetection()  # Updated class name
    rclpy.spin(cam_rgb_lidar_detection)
    cam_rgb_lidar_detection.destroy_node()
    rclpy.shutdown()

if _name_ == '_main_':
    main()