#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
import copy
import numpy as np
import cv2
import torch
import ast

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image as IMG
from image2world.image2worldlib import *
from fbot_recognition import BaseRecognition

from std_msgs.msg import Header, String
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from fbot_vision_msgs.msg import Detection3D, Detection3DArray
from vision_msgs.msg import BoundingBox2D, BoundingBox3D

from ament_index_python.packages import get_package_share_directory
class MoondreamRecognition(BaseRecognition):
    def __init__(self) -> None:
        super().__init__(nodeName='moondream_recognition')

        self.labels_dict: dict = {}
        self.current_class: str = ""
        self.declareParameters()
        self.readParameters()
        self.loadModel()
        self.initRosComm()

    def initRosComm(self) -> None:
        self.debugPublisher = self.create_publisher(Image, self.debugImageTopic, qos_profile=self.debugQosProfile)
        self.markerPublisher = self.create_publisher(MarkerArray, 'fbot_vision/fr/moondream_markers', qos_profile=self.debugQosProfile)
        self.objectRecognitionPublisher = self.create_publisher(Detection3DArray, self.objectRecognitionTopic, qos_profile=self.objectRecognitionQosProfile)
        self.objectPromptSubscriber = self.create_subscription(String, self.objectPromptTopic, qos_profile=self.qosProfile, callback=self.updateObjectPrompt)
        super().initRosComm(callbackObject=self)

    def loadModel(self) -> None: 
        self.get_logger().info("=> Loading model")
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
            # Uncomment to run on GPU.
            device_map={"": "cuda"}
        )
        self.get_logger().info("=> Loaded")

    def unLoadModel(self) -> None:
        del self.model
        torch.cuda.empty_cache()
        self.model = None

    def updateObjectPrompt(self, msg: String):
        self.current_class = msg.data

    def callback(self, depthMsg: Image, imageMsg: Image, cameraInfoMsg: CameraInfo) -> None:

        if self.current_class == "":
            self.get_logger().info("Waiting for object prompt to be set ...")
            return

        if imageMsg is None or depthMsg is None or cameraInfoMsg is None:
            self.get_logger().error("One or more input messages are invalid.")
            return
        
        cvImage = self.cvBridge.imgmsg_to_cv2(imageMsg,desired_encoding='bgr8')
        pilImage = IMG.fromarray(cvImage[..., ::-1])
        encImage = self.model.encode_image(pilImage)
        label = self.current_class
        results = self.model.detect(encImage, label)["objects"]

        detectionHeader = imageMsg.header

        detection3DArray = Detection3DArray()
        detection3DArray.header = detectionHeader
        detection3DArray.image_rgb = imageMsg

        if len(results):
            for box in results: 
                                
                score = 1.0

                bb2d = BoundingBox2D()
                data = BoundingBoxProcessingData()
                data.sensor.setSensorData(cameraInfoMsg, depthMsg)
                
                x_min = int(box['x_min'] * pilImage.width)
                x_max = int(box['x_max'] * pilImage.width)

                y_min = int(box['y_min'] * pilImage.height)
                y_max = int(box['y_max'] * pilImage.height)

                centerX = float((x_max + x_min)/2.0)
                centerY = float((y_max + y_min)/2.0)

                sizeX = float(x_max - x_min)
                sizeY = float(y_max - y_min)

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
                
                detection3d = self.createDetection3d(bb2d, bb3d, score, detectionHeader, label)
                if detection3d is not None:
                    detection3DArray.detections.append(detection3d)
                    
        self.objectRecognitionPublisher.publish(detection3DArray)
        self.labels_dict.clear()

        imageArray = cvImage.copy()
        if len(results):
            for box in results:
                x_min = int(box['x_min'] * pilImage.width)
                x_max = int(box['x_max'] * pilImage.width)

                y_min = int(box['y_min'] * pilImage.height)
                y_max = int(box['y_max'] * pilImage.height)

                imageArray = cv2.rectangle(imageArray, (x_min, y_min), (x_max, y_max), (255, 0, 255))
        image = IMG.fromarray(imageArray[..., ::-1])
        debugImageMsg = self.cvBridge.cv2_to_imgmsg(np.array(image), encoding='rgb8')
        self.debugPublisher.publish(debugImageMsg)

        self.publishMarkers(detection3DArray.detections)

    def createDetection3d(self, bb2d: BoundingBox2D, bb3d: BoundingBox3D , score: float, detectionHeader: Header, label: str) -> Detection3D:
        detection3d = Detection3D()
        detection3d.header = detectionHeader
        detection3d.score = score

        if '/' in label:
            detection3d.label = label
        else:
            detection3d.label = f"none/{label}" if label[0].islower() else f"None/{label}"

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
        
        self.markerPublisher.publish(markers)

    def declareParameters(self) -> None:
        self.declare_parameter("publishers.debug.topic", "/fbot_vision/fr/debug")
        self.declare_parameter("publishers.debug.qos_profile", 1)
        self.declare_parameter("publishers.object_recognition.topic", "/fbot_vision/fr/object_recognition")
        self.declare_parameter("publishers.object_recognition.qos_profile", 1)
        self.declare_parameter("max_sizes", [0.05, 0.05, 0.05])
        self.declare_parameter("subscribers.object_prompt", "/fbot_vision/fr/object_prompt")
        super().declareParameters()

    def readParameters(self) -> None:
        self.debugImageTopic = self.get_parameter("publishers.debug.topic").value
        self.debugQosProfile = self.get_parameter("publishers.debug.qos_profile").value
        self.objectRecognitionTopic = self.get_parameter("publishers.object_recognition.topic").value
        self.objectRecognitionQosProfile = self.get_parameter("publishers.object_recognition.qos_profile").value
        self.maxSizes = self.get_parameter("max_sizes").value
        self.objectPromptTopic = self.get_parameter("subscribers.object_prompt").value
        super().readParameters()

def main(args=None):
    rclpy.init(args=args)
    node = MoondreamRecognition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()