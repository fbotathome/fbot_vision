#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.clock import Clock
from PIL import Image as IMG

from fbot_recognition import BaseRecognition
import numpy as np
import os
from copy import copy
import cv2
from ultralytics import YOLO
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from ament_index_python.packages import get_package_share_directory
from fbot_vision_msgs.msg import Description2D, Recognitions2D
import torch

#TODO: Allocate and deallocate model in the right way

class YoloV8Recognition(BaseRecognition):
    def __init__(self, state=True):
        super().__init__(state=state)

        self.readParameters()

        self.colors = dict([(k, np.random.randint(low=0, high=256, size=(3,)).tolist()) for k in self.classes])

        self.loadModel()
        self.initRosComm()

    def initRosComm(self):
        self.debug_publisher = self.create_publisher(self.debug_topic, Image, queue_size=self.debug_qs)
        self.object_recognition_publisher = self.create_publisher(self.object_recognition_topic, Recognitions2D, queue_size=self.object_recognition_qs)
        self.people_detection_publisher = self.create_publisher(self.people_detection_topic, Recognitions2D, queue_size=self.people_detection_qs)
        super().initRosComm(callbacks_obj=self)

    def loadModel(self): 
        self.model = YOLO(self.model_file)
        self.model.conf = self.threshold
        print('Done loading model!')

    def unLoadModel(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = None

    def callback(self, *args):
        source_data = self.sourceDataFromArgs(args)

        if 'image_rgb' not in source_data:
            self.getlogger().warn('Souce data has no image_rgb.')
            return None
        
        img_rgb = source_data['image_rgb']
        self.getlogger().info('Image ID: ' + str(img_rgb.header.seq))

        cv_img = self.cv_bridge.imgmsg_to_cv2(img_rgb,desired_encoding='bgr8')
        results = self.model(cv_img)
        for r in results:
            im_array = r.plot()
            im = IMG.fromarray(im_array[..., ::-1])
        img_writen = self.cv_bridge.cv2_to_imgmsg(np.array(im), encoding='rgb8')
        self.debug_publisher.publish(img_writen)

    def readParameters(self):
        self.debug_topic = self.get_parameter("~publishers/debug/topic", "/butia_vision/br/debug")
        self.debug_qs = self.get_parameter("~publishers/debug/queue_size", 1)

        self.object_recognition_topic = self.get_parameter("~publishers/object_recognition/topic", "/butia_vision/br/object_recognition")
        self.object_recognition_qs = self.get_parameter("~publishers/object_recognition/queue_size", 1)

        self.people_detection_topic = self.get_parameter("~publishers/people_detection/topic", "/butia_vision/br/people_detection")
        self.people_detection_qs = self.get_parameter("~publishers/people_detection/queue_size", 1)

        self.threshold = self.get_parameter("~threshold", 0.5)


        self.all_classes = list(self.get_parameter("~all_classes", []))
        self.classes_by_category = dict(self.get_parameter("~classes_by_category", {}))
    
        self.model_file = get_package_share_directory('fbot_recognition') + "/weigths/" + self.get_parameter("~model_file", "yolov8n.pt")

        max_sizes = self.get_parameter("~max_sizes", [[0.05, 0.05, 0.05]])

        if len(max_sizes) == 1:
            max_sizes = [max_sizes[0] for _ in self.all_classes]

        if len(max_sizes) != len(self.all_classes):
            self.get_logger().error('Wrong definition of max_sizes in yaml of recognition node.')
            rclpy.shutdown()
            assert False
        
        self.max_sizes = dict([(self.all_classes[i], max_sizes[i]) for i in range(len(max_sizes))])

        super().readParameters()

def main(args=None):
    rclpy.init(args=args)
    node = YoloV8Recognition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()