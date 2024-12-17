#!/usr/bin/env python

import rclpy
import torch

from time import perf_counter

from ultralytics import YOLO
from ReIDManager import ReIDManager
from fbot_recognition import BaseRecognition
from vision_msgs.msg import Detection2D, Detection2DArray, Detection3D, Detection3DArray
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Empty
from ament_index_python.packages import get_package_share_directory

class YoloTrackerRecognition(BaseRecognition):
    def __init__(self, node_name):
        super().__init__(nodeName=node_name)
        self.tracking = False
        self.reid_manager = None
        self.declareParameters()
        self.readParameters()
        self.loadModel()
        self.initRosComm()
        if self.tracking_on_init:
            self.startTracking()
        # self.loadModel()

    def initRosComm(self):
        super().initRosComm(self)
        self.debugPub = self.create_publisher(Image, self.debug_topic, qos_profile=1)
        self.recognitionPub = self.create_publisher(Detection2DArray, self.recognition_topic, qos_profile=1)
        self.trackingPub = self.create_publisher(Detection2DArray, self.tracking_topic, qos_profile=1)
        
        self.trackingStartService = self.create_service(Empty, self.start_tracking_topic, self.startTracking)
        self.trackingStopService = self.create_service(Empty, self.stop_tracking_topic, self.stopTracking)
    
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
        return Empty()
    
    def stopTracking(self, req : Empty.Request, resp : Empty.Response):
        self.tracking = False
        self.unLoadTrackerModel()
        self.get_logger().info("Tracking stoped!!!")

    def callback(self, depthMsg: Image, imageMsg: Image, cameraInfoMsg: CameraInfo):
        self.get_logger().warn("Message arrived")

    def declareParameters(self):
        super().declareParameters()
        self.declare_parameter("publishers.debug.topic","/fbot_vision/br/debug")
        self.declare_parameter("publishers.pose_recognition.queue_size",1)
        self.declare_parameter("publishers.recognition.topic", "/fbot_vision/br/recognition")
        self.declare_parameter("publishers.recognition.queue_size", 1)

        self.declare_parameter("services.tracking.start","/fbot_vision/pt/start")
        self.declare_parameter("services.tracking.stop","/fbot_vision/pt/stop")
        
        self.declare_parameter("publishers.tracking.topic", "/fbot_vision/pt/tracking2D")
        self.declare_parameter("publishers.tracking.queue_size", 1)

        self.declare_parameter("debug_kpt_threshold", 0.5)

        self.declare_parameter("model_file","yolov8n-pose")
        self.declare_parameter("tracking.model_file","resnet_reid_model.pt")
        self.declare_parameter("tracking.model_name","resnet50.pt")

        self.declare_parameter("tracking.thresholds.det_threshold", 0.5)
        self.declare_parameter("tracking.thresholds.reid_threshold", 0.75)
        self.declare_parameter("tracking.thresholods.reid_threshold_feature_add",0.7)
        self.declare_parameter('tracking.thresholds.iou_threshold',0.5)
        self.declare_parameter("tracking.thresholds.max_time",60)
        self.declare_parameter("tracking.thresholds.max_age",5)
        self.declare_parameter("tracking.start_on_init", False)

        self.declare_parameter("tracking.reid_img_size.height",256)
        self.declare_parameter("tracking.reid_img_size.width",128)

        # self.declare_parameter("tracking.model_name",128)

        return 
    
    def readParameters(self):
        super().readParameters()
        self.debug_topic = self.get_parameter("publishers.debug.topic").value
        print(f"VALOR: {self.debug_topic}")
        self.debug_qs = self.get_parameter("publishers.pose_recognition.queue_size").value

        self.recognition_topic = self.get_parameter("publishers.recognition.topic").value
        self.recognition_qs = self.get_parameter("publishers.recognition.queue_size").value

        self.start_tracking_topic = self.get_parameter("services.tracking.start").value
        self.stop_tracking_topic = self.get_parameter("services.tracking.stop").value
        
        self.tracking_topic = self.get_parameter("publishers.tracking.topic").value
        self.tracking_qs = self.get_parameter("publishers.tracking.queue_size").value

        self.threshold = self.get_parameter("debug_kpt_threshold").value

        share_directory = get_package_share_directory("fbot_recognition")

        self.model_file = share_directory + "/weigths/" + self.get_parameter("model_file").value
        self.reid_model_file = share_directory + "/weigths/" + self.get_parameter("tracking.model_file").value
        self.reid_model_name = self.get_parameter("tracking.model_name").value

        self.det_threshold = self.get_parameter("tracking.thresholds.det_threshold").value
        self.reid_threshold = self.get_parameter("tracking.thresholds.reid_threshold").value
        self.reid_add_feature_threshold = self.get_parameter("tracking.thresholods.reid_threshold_feature_add").value
        self.iou_threshold = self.get_parameter('tracking.thresholds.iou_threshold').value
        self.max_time = self.get_parameter("tracking.thresholds.max_time").value
        self.max_age = self.get_parameter("tracking.thresholds.max_age").value
        self.tracking_on_init = self.get_parameter("tracking.start_on_init").value

        self.reid_img_size = (self.get_parameter("tracking.reid_img_size.height").value,self.get_parameter("tracking.reid_img_size.width").value)
    

    

def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloTrackerRecognition("yolo_tracker_recognition")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()