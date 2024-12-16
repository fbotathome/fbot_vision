#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy


import numpy as np
import os
import cv2
import face_recognition
import time
from ament_index_python.packages import get_package_share_directory
from collections import Counter
import pickle

from fbot_recognition import BaseRecognition

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from fbot_vision_msgs.msg import Detection3D, Detection3DArray
from vision_msgs.msg import BoundingBox2D, BoundingBox3D
from fbot_vision_msgs.srv import PeopleIntroducing
from geometry_msgs.msg import Vector3

from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor

import rclpy.wait_for_message

class FaceRecognition(BaseRecognition):
    def __init__(self):
        super().__init__(packageName='fbot_recognition', nodeName='face_recognition')
        packagePath = get_package_share_directory('fbot_recognition')
        datasetPath = os.path.join(packagePath, 'dataset')
        self.featuresPath = os.path.join(datasetPath, 'features')
        self.peopleDatasetPath = os.path.join(datasetPath, 'people/')
        self.declareParameters()
        self.readParameters()
        self.loadModel()
        self.initRosComm()

        knownFacesDict = self.loadVar('features')
        self.knownFaces = self.flatten(knownFacesDict)

        self.lastImage = None

    def initRosComm(self):
        self.groupTopic = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.groupServer = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        # print(self.debugImageTopic, self.faceRecognitionTopic, self.introducePersonServername)
        self.debugPublisher = self.create_publisher(Image, self.debugImageTopic, qos_profile=self.debugQosProfile)
        self.faceRecognitionPublisher = self.create_publisher(Detection3DArray, self.faceRecognitionTopic,  qos_profile=self.faceRecognitionQosProfile, callback_group=self.groupTopic)
        self.introducePersonService = self.create_service(PeopleIntroducing, self.introducePersonServername, self.peopleIntroducingCB, callback_group=self.groupServer)
        super().initRosComm(callbackObject=self)

    def loadModel(self):
        pass

    def unLoadModel(self):
        pass

    def callback(self, depthMsg: Image, imageMsg: Image, cameraInfoMsg: CameraInfo):

        self.lastImage = imageMsg

        threshold = 0.5
        faceRecognitions = Detection3DArray()
        # sourceData = self.sourceDataFromArgs(args)

        # h.seq = self.seq
        # self.seq += 1
        # h.stamp = rospy.Time.now()
        # h = Header()
        faceRecognitions.header = imageMsg.header
        faceRecognitions.image_rgb = imageMsg 
        
        #rospy.loginfo('Image ID: ' + str(img.header.seq))

        cvImage = self.cvBridge.imgmsg_to_cv2(imageMsg)

        currentFaces = face_recognition.face_locations(cvImage, model = 'yolov8')
        currentFacesEncoding = face_recognition.face_encodings(cvImage, currentFaces)

        debugImg = cvImage
        names = []
        nameDistance=[]
        for idx in range(len(currentFacesEncoding)):
            currentEncoding = currentFacesEncoding[idx]
            top, right, bottom, left = currentFaces[idx]
            detection = Detection3D()
            name = 'unknown'
            if(len(self.knownFaces[0]) > 0):
                faceDistances = np.linalg.norm(self.knownFaces[1] - currentEncoding, axis = 1)
                faceDistanceMinIndex = np.argmin(faceDistances)
                minDistance = faceDistances[faceDistanceMinIndex]

                if minDistance < threshold:
                    name = (self.knownFaces[0][faceDistanceMinIndex])
            detection.label = name

            names.append(name)

            detectionHeader = imageMsg.header

            detection.header = detectionHeader
            # detection.type = Description2D.DETECTION
            # detection.id = description.header.seq
            # detection.score = 1
            # detection.max_size = Vector3(*[0.2, 0.2, 0.2])
            size = int(right-left), int(bottom-top)
            print(detection.bbox2d.center)
            detection.bbox2d.center.position.x = float(int(left) + int(size[1]/2))
            detection.bbox2d.center.position.y = float(int(top) + int(size[0]/2))
            detection.bbox2d.size_x = float(bottom-top)
            detection.bbox2d.size_y = float(right-left)

            cv2.rectangle(debugImg, (left, top), (right, bottom), (0, 255, 0), 2)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(debugImg, name, (left + 4, bottom - 4), font, 0.5, (0,0,255), 2)
            # description_header.seq += 1

            faceRecognitions.detections.append(detection)
        print(type(self.cvBridge.cv2_to_imgmsg(np.array(debugImg), encoding='rgb8')))
        self.debugPublisher.publish(self.cvBridge.cv2_to_imgmsg(np.array(debugImg), encoding='rgb8'))

        if len(faceRecognitions.detections) > 0:
            self.faceRecognitionPublisher.publish(faceRecognitions)

    def declareParameters(self):
        self.declare_parameter("publishers.debug.qos_profile", 1)
        self.declare_parameter("publishers.debug.topic", "/fbot_vision/fr/debug_face")
        self.declare_parameter("publishers.face_recognition.qos_profile", 1)
        self.declare_parameter("publishers.face_recognition.topic", "/fbot_vision/fr/face_recognition")
        self.declare_parameter("servers.introduce_person.servername", "/fbot_vision/fr/introduce_person")
        self.declare_parameter('model_path', 'weights/face_recognition/face_recognition.pth')
        super().declareParameters()

    def readParameters(self):
        self.debugImageTopic = self.get_parameter("publishers.debug.topic").value
        self.debugQosProfile = self.get_parameter("publishers.debug.qos_profile").value
        self.faceRecognitionTopic = self.get_parameter("publishers.face_recognition.topic").value
        self.faceRecognitionQosProfile = self.get_parameter("publishers.face_recognition.qos_profile").value
        self.introducePersonServername = self.get_parameter("servers.introduce_person.servername").value


        super().readParameters()
        self.modelPath = self.pkgPath + '/' + self.get_parameter('model_path').value

    def regressiveCounter(self, sec):
        sec = int(sec)
        for i  in range(0, sec):
            print(str(sec-i) + '...')
            time.sleep(1)

    def saveVar(self, variable, filename):
        with open(self.featuresPath + '/' +  filename + '.pkl', 'wb') as file:
            pickle.dump(variable, file)

    def loadVar(self, filename):
        filePath = self.featuresPath + '/' +  filename + '.pkl'
        if os.path.exists(filePath):
            with open(filePath, 'rb') as file:
                variable = pickle.load(file)
            return variable
        return {}

    def flatten(self, l):
        valuesList = [item for sublist in l.values() for item in sublist]
        keysList = [item for name in l.keys() for item in [name]*len(l[name])]
        return keysList, valuesList

    def encodeFaces(self):

        encodings = []
        names = []
        try:
            encodedFace = self.loadVar('features')
        except:
            encodedFace = {}
        trainDir = os.listdir(self.peopleDatasetPath)

        for person in trainDir:
            if person not in self.knownFaces[0]:   
                pix = os.listdir(self.peopleDatasetPath + person)

                for personImg in pix:
                    face = face_recognition.load_image_file(self.peopleDatasetPath + person + "/" + personImg)
                    faceBBs = face_recognition.face_locations(face, model = 'yolov8')

                    largestFace = None
                    largestArea = -float('inf')
                    for top, right, bottom, left in faceBBs:
                        area = (bottom - top)*(right - left)
                        if area > largestArea:
                            largestArea = area
                            largestFace = (top, right, bottom, left)

                    if largestFace is not None:
                        faceEncoding = face_recognition.face_encodings(face, known_face_locations=[largestFace])[0]
                        encodings.append(faceEncoding)

                        if person not in names:
                            names.append(person)
                            encodedFace[person] = []
                            encodedFace[person].append(faceEncoding)
                        else:
                            encodedFace[person].append(faceEncoding)
                    else:
                        print(person + "/" + personImg + " was skipped and can't be used for training")
            else:
                pass
        self.saveVar(encodedFace, 'features')             


    def peopleIntroducingCB(self, peopleIntroducingRequest, peopleIntroducingResponse):
        name = peopleIntroducingRequest.name
        numImages = peopleIntroducingRequest.num_images
        dirName = os.path.join(self.peopleDatasetPath, name)
        os.makedirs(dirName, exist_ok=True)
        os.makedirs(self.featuresPath, exist_ok=True)
        imageType = '.jpg'

        imageLabels = os.listdir(dirName)
        addImageLabels = []
        i = 1
        k = 0
        j = numImages
        number = []

        for label in imageLabels:
            self.get_logger().warning(label)
            number.append(int(float(label.replace(imageType, ''))))
        
        number.sort()
        n = 1
        while j > 0:
            if k < len(number):
                n = number[k] + 1
                if number[k] == i:
                    k += 1
                else:
                    addImageLabels.append((str(i) + imageType))
                    j -= 1      
                i += 1 

            else:
                addImageLabels.append(str(n) + imageType)
                j -= 1
                n += 1

        i = 0
        while i < numImages:
            self.regressiveCounter(peopleIntroducingRequest.interval)
            # try:
            # _,image = rclpy.wait_for_message.wait_for_message(Image, self, 'camera/camera/color/image_raw')
            image = self.lastImage
            # self.get_logger().warning(image[0])
            cvImage = self.cvBridge.imgmsg_to_cv2(image)
            # self.get_logger().warning(self.lastImage.header)
            
            faceLocations = face_recognition.face_locations(cvImage, model='yolov8')
                
            if len(faceLocations) > 0:
                cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(dirName, addImageLabels[i]), cvImage)
                self.get_logger().warning('Picture ' + addImageLabels[i] + ' was  saved.')
                i+= 1
            else:
                self.get_logger().warning("The face was not detected.")

        cv2.destroyAllWindows()
        
        peopleIntroducingResponse.response = True

        self.encodeFaces()

        knownFacesDict = self.loadVar('features')
        self.knownFaces = self.flatten(knownFacesDict)
        return peopleIntroducingResponse


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognition()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    executor.spin()
    # rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
