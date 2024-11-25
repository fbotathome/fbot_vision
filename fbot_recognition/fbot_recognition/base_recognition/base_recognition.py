#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image
from std_srvs.srv import Empty

from fbot_vision_msgs.srv import ListClasses, ListClassesResponse

from fbot_vision_bridge import VisionSynchronizer

class BaseRecognition(Node):
    def __init__(self, state=True, package_name='fbot_recognition'):
        self.state = state
        self.pkg_path = get_package_share_directory(package_name)
        self.seq = 0

    @staticmethod
    def addSourceData2Recognitions2D(source_data, recognitions2d):
        for key, value in source_data.items():
            setattr(recognitions2d, key, value)
        return recognitions2d
    
    def initRosComm(self, callbacks_obj=None):
        if callbacks_obj is None:
            callbacks_obj = self
        VisionSynchronizer.syncSubscribers(self.subscribers_dict, callbacks_obj.callback, queue_size=self.queue_size, exact_time=self.exact_time, slop=self.slop)
        self.list_classes_server = self.create_service(self.list_classes_service, ListClasses, callbacks_obj.serverListClasses)

    def sourceDataFromArgs(self, args):
        data = {}
        index = 0
        for source in VisionSynchronizer.POSSIBLE_SOURCES:
            if source in self.subscribers_dict:
                data[source] = args[index]
                index+= 1
        return data

    def serverListClasses(self, req):
        return ListClassesResponse(self.classes)

    def loadModel(self):
        pass

    def unLoadModel(self):
        pass

    def callback(self, *args):
        pass

    def readParameters(self):
        self.subscribers_dict = dict(self.get_parameter("~subscribers", {}))
        self.queue_size = self.subscribers_dict.pop('queue_size', 1)
        self.exact_time = self.subscribers_dict.pop('exact_time', False)
        self.slop = self.subscribers_dict.pop('slop', 0.1)

        self.list_classes_service = self.get_parameter("~servers/list_classes/service", "/fbot_vision/bgr/list_classes")

        self.classes = list(self.get_parameter("~classes", [])) 