#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
import copy
import numpy as np
import torch

from ultralytics import YOLO
from PIL import Image as IMG
from image2world.image2worldlib import *
from fbot_recognition import BaseRecognition
from rclpy.lifecycle import LifecycleNode

from std_msgs.msg import Header
from std_srvs.srv import Empty
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from fbot_vision_msgs.msg import Detection3D, Detection3DArray
from vision_msgs.msg import BoundingBox2D, BoundingBox3D

from ament_index_python.packages import get_package_share_directory

#TODO: Filter the area inside the house by using i2w.inPolygonFilter()

class YoloV8Recognition(BaseRecognition):
    def __init__(self) -> None:
        super().__init__(nodeName='yolov8_recognition')

        self.labels_dict: dict = {}
        self.model = None
        self.run = False
        self.declareParameters()
        self.readParameters()
        self.initRosComm()
        if self.start_on_init:
            self._startRecognition()

    def initRosComm(self) -> None:
        self.debugPublisher = self.create_publisher(Image, self.debugImageTopic, qos_profile=self.debugQosProfile)
        self.markerPublisher = self.create_publisher(MarkerArray, 'fbot_vision/fr/object_markers', qos_profile=self.debugQosProfile)
        self.objectRecognitionPublisher = self.create_publisher(Detection3DArray, self.objectRecognitionTopic, qos_profile=self.objectRecognitionQosProfile)
        self.recognitionStartService = self.create_service(Empty, self.startRecognitionTopic, self.startRecognition)
        self.recognitionStopService = self.create_service(Empty, self.stopRecognitionTopic, self.stopRecognition)
        super().initRosComm(callbackObject=self)

    def loadModel(self) -> None: 
        self.get_logger().info("=> Loading model")
        self.model = YOLO(self.modelFile)
        self.model.conf = self.threshold
        self.get_logger().info("=> Loaded")

    def unLoadModel(self) -> None:
        del self.model
        torch.cuda.empty_cache()
        self.model = None

    def _startRecognition(self):
        self.loadModel()
        self.run = True
        self.get_logger().info("Starting Object Recognition!!!")

    def _stopRecognition(self):
        self.run = False
        self.unLoadModel()
        self.get_logger().info("Stopping Object Recognition!!!")

    def startRecognition(self, req: Empty.Request, resp: Empty.Response):
        self._startRecognition()
        return resp

    def stopRecognition(self, req: Empty.Request, resp: Empty.Response):
        self._stopRecognition()
        return resp

    def callback(self, depthMsg: Image, imageMsg: Image, cameraInfoMsg: CameraInfo) -> None:

        if not self.run:
            return
        
        if self.model is None:
            self.get_logger().error("Model is not loaded.")
            return

        if imageMsg is None or depthMsg is None or cameraInfoMsg is None:
            self.get_logger().error("One or more input messages are invalid.")
            return
        
        cvImage = self.cvBridge.imgmsg_to_cv2(imageMsg,desired_encoding='bgr8')
        results = self.model(cvImage)

        detectionHeader = imageMsg.header

        detection3DArray = Detection3DArray()
        detection3DArray.header = detectionHeader
        detection3DArray.image_rgb = imageMsg

        if len(results[0].boxes):
            for box in results[0].boxes: 

                if box is None:
                    return None
                
                classId = int(box.cls)
                
                label = results[0].names[classId]
                score = float(box.conf)

                bb2d = BoundingBox2D()
                data = BoundingBoxProcessingData()
                data.sensor.setSensorData(cameraInfoMsg, depthMsg)

                centerX, centerY, sizeX, sizeY = map(float, box.xywh[0])

                data.boundingBox2D.center.position.x = centerX
                data.boundingBox2D.center.position.y = centerY
                data.boundingBox2D.size_x = sizeX
                data.boundingBox2D.size_y = sizeY
                data.maxSize.x = self.maxSizes[0]
                data.maxSize.y = self.maxSizes[1]
                data.maxSize.z = self.maxSizes[2]

                bb2d = data.boundingBox2D
        
                try:
                    bb3d = boundingBoxProcessing(data)
                except Exception as e:
                    self.get_logger().error(f"Error processing bounding box: {e}")
                    continue

                if np.linalg.norm([bb3d.center.position.x,bb3d.center.position.y,bb3d.center.position.z]) < 0.05:
                    continue   
                
                detection3d = self.createDetection3d(bb2d, bb3d, score, detectionHeader, label)
                if detection3d is not None:
                    detection3DArray.detections.append(detection3d)
                    
        self.objectRecognitionPublisher.publish(detection3DArray)
        self.labels_dict.clear()

        imageArray = results[0].plot()
        image = IMG.fromarray(imageArray[..., ::-1])
        debugImageMsg = self.cvBridge.cv2_to_imgmsg(np.array(image), encoding='rgb8')
        self.debugPublisher.publish(debugImageMsg)

        self.publishMarkers(detection3DArray.detections)

    def createDetection3d(self, bb2d: BoundingBox2D, bb3d: BoundingBox3D , score: float, detectionHeader: Header, label: str) -> Detection3D:
        detection3d = Detection3D()
        detection3d.header = detectionHeader
        detection3d.score = score

        if '-' in label:
            detection3d.label = label
        else:
            detection3d.label = f"none-{label}" if label[0].islower() else f"None-{label}"

        if detection3d.label in self.labels_dict:
            self.labels_dict[detection3d.label] += 1
        else:
            self.labels_dict[detection3d.label] = 1
            
        detection3d.id = self.labels_dict[detection3d.label]

        detection3d.bbox2d = copy.deepcopy(bb2d)
        detection3d.bbox3d = bb3d

        return detection3d


    def publishMarkers(self, descriptions3d) -> None:
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
        
        self.markerPublisher.publish(markers)

    def declareParameters(self) -> None:
        self.declare_parameter("publishers.debug.topic", "/fbot_vision/fr/debug")
        self.declare_parameter("publishers.debug.qos_profile", 1)
        self.declare_parameter("publishers.object_recognition.topic", "/fbot_vision/fr/object_recognition")
        self.declare_parameter("publishers.object_recognition.qos_profile", 1)
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("model_file", "yolov8n.pt")
        self.declare_parameter("max_sizes", [0.05, 0.05, 0.05])
        self.declare_parameter("start_on_init", True)
        self.declare_parameter("services.object_recognition.start", "/fbot_vision/fr/object_start")
        self.declare_parameter("services.object_recognition.stop", "/fbot_vision/fr/object_stop")
        super().declareParameters()

    def readParameters(self) -> None:
        self.debugImageTopic = self.get_parameter("publishers.debug.topic").value
        self.debugQosProfile = self.get_parameter("publishers.debug.qos_profile").value
        self.objectRecognitionTopic = self.get_parameter("publishers.object_recognition.topic").value
        self.objectRecognitionQosProfile = self.get_parameter("publishers.object_recognition.qos_profile").value
        self.threshold = self.get_parameter("threshold").value
        self.get_logger().info(f"Threshold: {self.threshold}")
        self.start_on_init = self.get_parameter("start_on_init").value
        self.modelFile = get_package_share_directory('fbot_recognition') + "/weights/" + self.get_parameter("model_file").value
        self.maxSizes = self.get_parameter("max_sizes").value
        self.startRecognitionTopic = self.get_parameter("services.object_recognition.start").value
        self.stopRecognitionTopic = self.get_parameter("services.object_recognition.stop").value
        super().readParameters()

def main(args=None):
    rclpy.init(args=args)
    node = YoloV8Recognition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()