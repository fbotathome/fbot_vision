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
    def __init__(self, packageName='fbot_recognition', nodeName='base_recognition'):
        super().__init__(nodeName)
        self.pkgPath = get_package_share_directory(packageName)
        self.topicsToSubscribe = {}

        self.loadCVBrige()
    
    def initRosComm(self, callbackObject=None):
        if callbackObject is None:
            callbackObject = self
        self.syncSubscribers(callbackObject.callback)

    def syncSubscribers(self, callbackObject):
        subscribers = []
        for source in SOURCES_TYPES:
            if source in self.topicsToSubscribe:
                subscribers.append(message_filters.Subscriber(self, SOURCES_TYPES[source], self.topicsToSubscribe[source], qos_profile=self.qosProfile))
        self._synchronizer = message_filters.ApproximateTimeSynchronizer([x for x in subscribers],queue_size=1, slop=self.slop)
        self._synchronizer.registerCallback(callbackObject)

    def loadCVBrige(self):
        self.cvBridge = CvBridge()

    def loadModel(self):
        pass

    def unLoadModel(self):
        pass

    def callback(self, *args):
        pass

    def readParameters(self):
        for source in SOURCES_TYPES:
            self.declare_parameter(f'subscribers.{source}', "")
            self.topicsToSubscribe[source] = self.get_parameter(f'subscribers.{source}').value 
        self.slop = self.topicsToSubscribe.pop('slop', 0.1)
        self.qosProfile = self.topicsToSubscribe.pop('qos_profile', 1)
    