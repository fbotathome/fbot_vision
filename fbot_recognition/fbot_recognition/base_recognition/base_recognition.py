#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
import message_filters

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge

SOURCES_TYPES = {
        'camera_info': CameraInfo,
        'image_rgb': Image,
        'image_depth': Image
    }
class BaseRecognition(Node):
    def __init__(self, package_name='fbot_recognition', node_name='base_recognition'):
        super().__init__(node_name)
        self.pkg_path = get_package_share_directory(package_name)
        self.subscribers_dict = {}

        self.loadCVBrige()
    
    def initRosComm(self, callback_obj=None):
        if callback_obj is None:
            callback_obj = self
        self.syncSubscribers(callback_obj.callback)

    def syncSubscribers(self, callback_obj):
        subscribers = []
        for source in SOURCES_TYPES:
            if source in self.subscribers_dict:
                subscribers.append(message_filters.Subscriber(self, SOURCES_TYPES[source], self.subscribers_dict[source], qos_profile=self.qos_profile))
        self._synchronizer = message_filters.ApproximateTimeSynchronizer([x for x in subscribers],queue_size=10, slop=self.slop)
        self._synchronizer.registerCallback(callback_obj)

    def loadCVBrige(self):
        self.cv_bridge = CvBridge()

    def loadModel(self):
        pass

    def unLoadModel(self):
        pass

    def callback(self, *args):
        pass

    def readParameters(self):
        for source in SOURCES_TYPES:
            self.declare_parameter(f'subscribers.{source}', "")
            self.subscribers_dict[source] = self.get_parameter(f'subscribers.{source}').value 
        self.slop = self.subscribers_dict.pop('slop', 0.1)
        self.qos_profile = self.subscribers_dict.pop('qos_profile', 1)
    