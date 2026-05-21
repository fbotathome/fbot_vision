#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

import rclpy
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image as IMG

from std_msgs.msg import Header
from std_srvs.srv import Empty
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import BoundingBox2D
from fbot_recognition import BaseRecognition
from fbot_vision_msgs.msg import Detection2D, Detection2DArray

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
        self.objectRecognitionPublisher = self.create_publisher(Detection2DArray, self.objectRecognitionTopic, qos_profile=self.objectRecognitionQosProfile)
        self.recognitionStartService = self.create_service(Empty, self.startRecognitionTopic, self.startRecognition)
        self.recognitionStopService = self.create_service(Empty, self.stopRecognitionTopic, self.stopRecognition)
        super().initRosComm(callbackObject=self)

    def loadModel(self) -> None: 
        self.get_logger().info("=> Loading model")
        self.model = YOLO(self.modelFile)
        self.model.conf = self.threshold
        self.get_logger().info("=> Loaded")

    def unloadModel(self) -> None:
        del self.model
        torch.cuda.empty_cache()
        self.model = None

    def _startRecognition(self):
        self.loadModel()
        self.run = True
        self.get_logger().info("Starting Object Recognition!!!")

    def _stopRecognition(self):
        self.run = False
        self.unloadModel()
        self.get_logger().info("Stopping Object Recognition!!!")

    def startRecognition(self, req: Empty.Request, resp: Empty.Response):
        self._startRecognition()
        return resp

    def stopRecognition(self, req: Empty.Request, resp: Empty.Response):
        self._stopRecognition()
        return resp

    def callback(self, cameraInfoMsg: CameraInfo, imageMsg: Image, depthMsg: Image) -> None:

        if not self.run:
            return
        
        if self.model is None:
            self.get_logger().error("Model is not loaded.")
            return

        if imageMsg is None or depthMsg is None or cameraInfoMsg is None:
            self.get_logger().error("One or more input messages are invalid.")
            return
        
        cvImage = self.cvBridge.imgmsg_to_cv2(imageMsg,desired_encoding='bgr8')
        results = self.model(cvImage, verbose=False)

        detectionHeader = imageMsg.header

        detection2DArray = Detection2DArray()
        detection2DArray.header = detectionHeader
        detection2DArray.image_rgb = imageMsg
        detection2DArray.image_depth = depthMsg
        detection2DArray.camera_info = cameraInfoMsg



        if len(results[0].boxes):
            for i ,box in enumerate(results[0].boxes): 

                if box is None:
                    return None
                
                mask = None
                if results[0].masks is not None:
                    mask = results[0].masks[i].data[0].cpu().numpy()
                
                classId = int(box.cls)
                
                label = results[0].names[classId]
                score = float(box.conf)

                box_coords = box.xyxy[0].cpu().numpy()

                box_msg = self.createDetection2d(box_coords, score, detectionHeader, label, i, -1, mask)
                detection2DArray.detections.append(box_msg)
                
                
                    
        self.objectRecognitionPublisher.publish(detection2DArray)
        self.labels_dict.clear()

        imageArray = results[0].plot()
        image = IMG.fromarray(imageArray[..., ::-1])
        debugImageMsg = self.cvBridge.cv2_to_imgmsg(np.array(image), encoding='rgb8')
        self.debugPublisher.publish(debugImageMsg)

    def createDetection2d(self, coord : np.ndarray, score: float, detectionHeader: Header, label: str, id : int, global_id: int, segmentation_mask : np.ndarray  = None) -> Detection2D:
        '''
        @brief Creates the detection2D messagem from the raw data.
        @param coord: Numpy array containing the coordinates of the bounding box on the format x1, y1, x2, y2.
        '''
        if coord.shape != (4,):
            raise ValueError(f"Expected coord shape of (4,), got {coord.shape}")

        msg = Detection2D()
        # msg.type |= Detection2D.DETECTION
        msg.header = detectionHeader
        msg.score = score
        msg.label = label
        msg.id = id

        msg.bbox.center.position.x = (coord[0]+coord[2])/2
        msg.bbox.center.position.y = (coord[1]+coord[3])/2
        msg.bbox.size_x = float(coord[2]-coord[0])
        msg.bbox.size_y = float(coord[3]-coord[1])
        
        msg.max_size.x = self.maxSizes[0]
        msg.max_size.y = self.maxSizes[1]
        msg.max_size.z = self.maxSizes[2]

        if segmentation_mask is not None:
            msg.mask = self.cvBridge.cv2_to_imgmsg(segmentation_mask, encoding="mono8")
            msg.type |= Detection2D.INSTANCE_SEGMENTATION        

        return msg


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
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("model_file", "yolo11x-seg.pt")
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