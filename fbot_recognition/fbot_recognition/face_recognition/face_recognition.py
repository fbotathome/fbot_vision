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
import copy
from image2world.image2worldlib import *

from fbot_recognition import BaseRecognition

import rclpy.logging
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from fbot_vision_msgs.msg import Detection3D, Detection3DArray
from vision_msgs.msg import BoundingBox2D, BoundingBox3D
from fbot_vision_msgs.srv import PeopleIntroducing
from geometry_msgs.msg import Vector3

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

    def initRosComm(self):
        self.debugPublisher = self.create_publisher(Image, self.debugImageTopic, qos_profile=self.debugQosProfile)
        self.faceRecognitionPublisher = self.create_publisher(Detection3DArray, self.faceRecognitionTopic,  qos_profile=self.faceRecognitionQosProfile)
        self.introducePersonService = self.create_service(PeopleIntroducing, self.introducePersonServername, self.peopleIntroducingCB)
        super().initRosComm(callbackObject=self)

    def loadModel(self):
        pass

    def unLoadModel(self):
        pass

    def callback(self, depthMsg: Image, imageMsg: Image, cameraInfoMsg: CameraInfo):
        try:
            faceRecognitions = Detection3DArray()
            faceRecognitions.header = imageMsg.header
            faceRecognitions.image_rgb = imageMsg 
        
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

                    if minDistance < self.threshold:
                        name = (self.knownFaces[0][faceDistanceMinIndex])
                detection.label = name

                names.append(name)

                detectionHeader = imageMsg.header

                detection.header = detectionHeader
                size = int(right-left), int(bottom-top)
                
                detection.bbox2d.center.position.x = float(int(left) + int(size[1]/2))
                detection.bbox2d.center.position.y = float(int(top) + int(size[0]/2))
                detection.bbox2d.size_x = float(bottom-top)
                detection.bbox2d.size_y = float(right-left)

                bb2d = BoundingBox2D()
                data = BoundingBoxProcessingData()
                data.sensor.setSensorData(cameraInfoMsg, depthMsg)


                data.boundingBox2D.center.position.x = float(int(left) + int(size[1]/2))
                data.boundingBox2D.center.position.y = float(int(top) + int(size[0]/2))
                data.boundingBox2D.size_x = float(bottom-top)
                data.boundingBox2D.size_y = float(right-left)
                data.maxSize.x = float(3)
                data.maxSize.y = float(3)
                data.maxSize.z = float(3)

                bb2d = data.boundingBox2D
        
                try:
                    bb3d = boundingBoxProcessing(data)
                except Exception as e:
                    self.get_logger().error(f"Error processing bounding box: {e}")
                    continue

                cv2.rectangle(debugImg, (left, top), (right, bottom), (0, 255, 0), 2)
                
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(debugImg, name, (left + 4, bottom - 4), font, 0.5, (0,0,255), 2)
                detection3d = self.createDetection3d(bb2d, bb3d, detectionHeader, name)
                if detection3d is not None:
                    faceRecognitions.detections.append(detection3d)

            self.debugPublisher.publish(self.cvBridge.cv2_to_imgmsg(np.array(debugImg), encoding='rgb8'))

            if len(faceRecognitions.detections) > 0:
                self.faceRecognitionPublisher.publish(faceRecognitions)
        except KeyError as e:
            while True:
                self.get_logger().warning(f"callback error {e}")

    def declareParameters(self):
        self.declare_parameter("publishers.debug.qos_profile", 1)
        self.declare_parameter("publishers.debug.topic", "/fbot_vision/fr/debug_face")
        self.declare_parameter("publishers.face_recognition.qos_profile", 1)
        self.declare_parameter("publishers.face_recognition.topic", "/fbot_vision/fr/face_recognition")
        self.declare_parameter("servers.introduce_person.servername", "/fbot_vision/fr/introduce_person")
        self.declare_parameter('model_path', 'weights/face_recognition/face_recognition.pth')
        self.declare_parameter("threshold", 0.8)
        super().declareParameters()

    def readParameters(self):
        self.debugImageTopic = self.get_parameter("publishers.debug.topic").value
        self.debugQosProfile = self.get_parameter("publishers.debug.qos_profile").value
        self.faceRecognitionTopic = self.get_parameter("publishers.face_recognition.topic").value
        self.faceRecognitionQosProfile = self.get_parameter("publishers.face_recognition.qos_profile").value
        self.introducePersonServername = self.get_parameter("servers.introduce_person.servername").value
        self.threshold = self.get_parameter("threshold").value


        super().readParameters()
        self.modelPath = self.pkgPath + '/' + self.get_parameter('model_path').value

    def regressiveCounter(self, sec):
        sec = int(sec)
        for i  in range(0, sec):
            self.get_logger().warning(str(sec-i) + '...')
            time.sleep(1)

    def saveVar(self, variable, filename):
        try:
            with open(self.featuresPath + '/' +  filename + '.pkl', 'wb') as file:
                pickle.dump(variable, file)
        except KeyError as e:
            while True:
                self.get_logger().warning(f"Save var error {e}")

    def loadVar(self, filename):
        try:
            filePath = self.featuresPath + '/' +  filename + '.pkl'
            if os.path.exists(filePath):
                with open(filePath, 'rb') as file:
                    variable = pickle.load(file)
                return variable
            return {}
        except KeyError as e:
            while True:
                self.get_logger().warning(f"LoadVar error {e}")

    def flatten(self, l):
        try:
            valuesList = [item for sublist in l.values() for item in sublist]
            keysList = [item for name in l.keys() for item in [name]*len(l[name])]
            return keysList, valuesList

        except KeyError as e:
            while True:
                self.get_logger().warning(f"Flatten error {e}")

    def encodeFaces(self, faceBoundingBoxes, faceImage):

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
                    # face = face_recognition.load_image_file(self.peopleDatasetPath + person + "/" + personImg)
                    # faceBBs = face_recognition.face_locations(face, model = 'yolov8')

                    largestFace = None
                    largestArea = -float('inf')
                    for top, right, bottom, left in faceBoundingBoxes:
                        area = (bottom - top)*(right - left)
                        if area > largestArea:
                            largestArea = area
                            largestFace = (top, right, bottom, left)

                    if largestFace is not None:
                        faceEncoding = face_recognition.face_encodings(faceImage, known_face_locations=[largestFace])[0]
                        encodings.append(faceEncoding)

                        if person not in names:
                            names.append(person)
                            encodedFace[person] = []
                            encodedFace[person].append(faceEncoding)
                        else:
                            encodedFace[person].append(faceEncoding)
                    else:
                        self.get_logger().warning(person + "/" + personImg + " was skipped and can't be used for training")
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

        faceBoundingBoxes=[]

        imageLabels = os.listdir(dirName)
        addImageLabels = []
        currentIndex = 1
        existingNumbers = [] 

        for label in imageLabels:
            existingNumbers.append(int(label.replace(imageType, '')))

        existingNumbers.sort()

        i = 0
        while len(addImageLabels) < numImages:
            if i < len(existingNumbers) and existingNumbers[i] == currentIndex:
                i += 1  
            else:
                addImageLabels.append(str(currentIndex) + imageType)  
            currentIndex += 1

        i = 0
        while i < numImages:
            self.regressiveCounter(peopleIntroducingRequest.interval)
            
            try:
                # _,image = rclpy.wait_for_message.wait_for_message(Image, self, self.topicsToSubscribe['image_rgb'])
                _,faceMessage = rclpy.wait_for_message.wait_for_message(Detection3DArray, self, self.faceRecognitionTopic)
                image = faceMessage.image_rgb
                cvImage = self.cvBridge.imgmsg_to_cv2(image)
                cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
            except (Exception) as e:
                break
            

            for faceInfos in faceMessage.detections:

                if faceInfos.label == 'unknown':
                    top = int(faceInfos.pose[0].y)
                    right = int(faceInfos.pose[1].x)
                    bottom = int(faceInfos.pose[1].y)
                    left = int(faceInfos.pose[0].x)
                    faceBoundingBoxes.append((top, right, bottom, left))

            if len(faceBoundingBoxes) > 0:

            # faceLocations = face_recognition.face_locations(cvImage, model='yolov8')
                
            # if len(faceLocations) > 0:
                
                cv2.imwrite(os.path.join(dirName, addImageLabels[i]), cvImage)
                self.get_logger().warning('Picture ' + addImageLabels[i] + ' was  saved.')
                i+= 1
            else:
                self.get_logger().warning("The face was not detected.")

        cv2.destroyAllWindows()
        
        peopleIntroducingResponse.response = True

        self.encodeFaces(faceBoundingBoxes, cvImage)

        knownFacesDict = self.loadVar('features')
        self.knownFaces = self.flatten(knownFacesDict)
        return peopleIntroducingResponse
    
    def createDetection3d(self, bb2d: BoundingBox2D, bb3d: BoundingBox3D, detectionHeader: Header, label: str) -> Detection3D:
        detection3d = Detection3D()
        detection3d.header = detectionHeader
        detection3d.id = 0
        detection3d.label = label
        detection3d.bbox2d = copy.deepcopy(bb2d)
        detection3d.bbox3d = copy.deepcopy(bb3d)

        return detection3d


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognition()
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
