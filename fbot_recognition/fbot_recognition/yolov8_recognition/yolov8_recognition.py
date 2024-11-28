#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from PIL import Image as IMG
from image2world.image2worldlib import *
from fbot_recognition import BaseRecognition
import numpy as np
import torch
import ast
from ultralytics import YOLO

from std_msgs.msg import Header
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker, MarkerArray
from fbot_vision_msgs.msg import Detection3D, Detection3DArray
from vision_msgs.msg import BoundingBox2D, BoundingBox3D

from ament_index_python.packages import get_package_share_directory




#TODO: Allocate and deallocate model in the right way
#TODO Implement people detection
#TODO: Filter the area inside the house by using i2w.inPolygonFilter()
#TODO: Need one declare parameters and one read parameters functions

class YoloV8Recognition(BaseRecognition):
    def __init__(self):
        super().__init__(node_name='yolov8_recognition')

        self.declareParameters()
        self.readParameters()
        self.loadModel()
        self.initRosComm()

    def initRosComm(self):
        self.debug_publisher = self.create_publisher(Image, self.debug_image_topic, qos_profile=self.debug_qp)
        self.marker_publisher = self.create_publisher(MarkerArray, 'pub/markers', qos_profile=self.debug_qp)
        self.object_recognition_publisher = self.create_publisher(Detection3DArray, self.object_recognition_topic, qos_profile=self.object_recognition_qp)
        self.people_detection_publisher = self.create_publisher(Detection3DArray, self.people_detection_topic, qos_profile=self.people_detection_qp)
        super().initRosComm(callback_obj=self)

    def loadModel(self): 
        self.get_logger().info("=> Loading model")
        self.model = YOLO(self.model_file)
        self.model.conf = self.threshold
        self.get_logger().info("=> Loaded")

    def unLoadModel(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = None

    def callback(self, depth_msg: Image, image_msg: Image, camera_info_msg: CameraInfo):

        self.get_logger().info("=> Entering callback ")

        all_classes = [cls for sublist in self.classes_by_category.values() for cls in sublist]
        all_classes_len = len(all_classes)

        if image_msg is None or depth_msg is None or camera_info_msg is None:
            self.get_logger().error("One or more input messages are invalid.")
            return
        
        cv_img = self.cv_bridge.imgmsg_to_cv2(image_msg,desired_encoding='bgr8')
        results = self.model(cv_img)

        detection_header = image_msg.header

        detection3darray = Detection3DArray()
        detection3darray.header = detection_header
        detection3darray.image_rgb = image_msg

        if len(results[0].boxes):
            for box in results[0].boxes: 

                if box is None:
                    return None
                
                bb2d = BoundingBox2D()
                class_id = int(box.cls)

                if class_id >= all_classes_len:
                    self.get_logger().error(f"Class id {class_id} not found in classes_by_category")
                    return
                
                label = results[0].names[class_id]
                score = float(box.conf)

                data = BoundingBoxProcessingData()
                data.sensor.setSensorData(camera_info_msg, depth_msg)

                center_x, center_y, size_x, size_y = map(float, box.xywh[0])

                data.boundingBox2D.center.position.x = center_x
                data.boundingBox2D.center.position.y = center_y
                data.boundingBox2D.size_x = size_x
                data.boundingBox2D.size_y = size_y
                data.maxSize.x = self.max_sizes[0]
                data.maxSize.y = self.max_sizes[1]
                data.maxSize.z = self.max_sizes[2]

                bb2d = data.boundingBox2D
        
                try:
                    bb3d = boundingBoxProcessing(data)
                except Exception as e:
                    self.get_logger().error(f"Error processing bounding box: {e}")
                    return None
                
                detection3d = self.createDetection3d(bb2d, bb3d, score, detection_header, label)
                if detection3d is not None:
                    detection3darray.detections.append(detection3d)
            
                ######PARA FILTRAR OQ ESTA DENTRO DA CASA
                #if i2w.inPolygonFilter(bbox3d):
        
        self.object_recognition_publisher.publish(detection3darray)

        im_array = results[0].plot()
        im = IMG.fromarray(im_array[..., ::-1])
        debug_imgmsg = self.cv_bridge.cv2_to_imgmsg(np.array(im), encoding='rgb8')
        self.debug_publisher.publish(debug_imgmsg)

        self.publishMarkers(detection3darray.detections)

    def createDetection3d(self, bb2d: BoundingBox2D, bb3d: BoundingBox3D , score: float, detection_header: Header, label: str):
        detection3d = Detection3D()
        detection3d.header = detection_header
        detection3d.id = 0
        detection3d.label == ''
        detection3d.score = score

        for category, items in self.classes_by_category.items():
            if label in items:
                detection3d.label = category + '/' + label
        if detection3d.label == '':
            self.get_logger().error(f"Label {label} not found in classes_by_category")
            return None

        detection3d.bbox2d = bb2d
        detection3d.bbox3d = bb3d

        return detection3d


    def publishMarkers(self, descriptions3d):
        markers = MarkerArray()
        duration = Duration()
        duration.sec = 2
        color = np.asarray([255, 0, 0])/255.0
        for i, det in enumerate(descriptions3d):
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
                if kpt3D.score > 0:
                    marker = Marker()
                    marker.header = det.header
                    marker.type = Marker.SPHERE
                    marker.id = idx
                    marker.color.r = color[1]
                    marker.color.g = color[2]
                    marker.color.b = color[0]
                    marker.color.a = 1
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
        
        self.marker_publisher.publish(markers)

    def declareParameters(self):
        self.declare_parameter("publishers.debug.topic", "/fbot_vision/fr/debug")
        self.declare_parameter("publishers.debug.qos_profile", 1)
        self.declare_parameter("publishers.object_recognition.topic", "/fbot_vision/fr/object_recognition")
        self.declare_parameter("publishers.object_recognition.qos_profile", 1)
        self.declare_parameter("publishers.people_detection.topic", "/fbot_vision/fr/people_detection")
        self.declare_parameter("publishers.people_detection.qos_profile", 1)
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("classes_by_category", "")
        self.declare_parameter("model_file", "yolov8n.pt")
        self.declare_parameter("max_sizes", [0.05, 0.05, 0.05])

    def readParameters(self):
        self.debug_image_topic = self.get_parameter("publishers.debug.topic").value
        self.debug_qp = self.get_parameter("publishers.debug.qos_profile").value
        self.object_recognition_topic = self.get_parameter("publishers.object_recognition.topic").value
        self.object_recognition_qp = self.get_parameter("publishers.object_recognition.qos_profile").value
        self.people_detection_topic = self.get_parameter("publishers.people_detection.topic").value
        self.people_detection_qp = self.get_parameter("publishers.people_detection.qos_profile").value
        self.threshold = self.get_parameter("threshold").value
        self.classes_by_category = ast.literal_eval(self.get_parameter('classes_by_category').value) #Need to use literal_eval since rclpy doesn't support dictionaries as a parameter
        self.model_file = get_package_share_directory('fbot_recognition') + "/weigths/" + self.get_parameter("model_file").value
        self.max_sizes = self.get_parameter("max_sizes").value
        super().readParameters()

def main(args=None):
    rclpy.init(args=args)
    node = YoloV8Recognition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()