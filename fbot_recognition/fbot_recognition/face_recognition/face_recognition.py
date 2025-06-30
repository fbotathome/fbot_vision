import rclpy

import os
import cv2
# import face_recognition
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
from rclpy.callback_groups import ReentrantCallbackGroup
import rclpy.wait_for_message

import numpy as np
import pandas as pd
import requests
import redis
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

# from deepface.commons import functions
from tqdm import tqdm
from deepface import DeepFace

from uuid import uuid4

class FaceRecognition(BaseRecognition):
    def __init__(self):
        super().__init__(packageName='fbot_recognition', nodeName='face_recognition')
        packagePath = get_package_share_directory('fbot_recognition')
        datasetPath = os.path.join(packagePath, 'dataset')
        self.featuresPath = os.path.join(datasetPath, 'features')
        self.peopleDatasetPath = os.path.join(datasetPath, 'people/')
        self.declareParameters()
        self.readParameters()
        self.initRosComm()
        
        self.configureRedis()


        knownFacesDict = self.loadVar('features')
        self.knownFaces = self.flatten(knownFacesDict)

    def initRosComm(self):
        self.debugPublisher = self.create_publisher(Image, self.debugImageTopic, qos_profile=self.debugQosProfile)
        self.faceRecognitionPublisher = self.create_publisher(Detection3DArray, self.faceRecognitionTopic,  qos_profile=self.faceRecognitionQosProfile)
        service_cb_group = ReentrantCallbackGroup()
        self.introducePersonService = self.create_service(srv_type=PeopleIntroducing, srv_name=self.introducePersonServername, callback=self.peopleIntroducingCB, callback_group=service_cb_group)
        self.last_detection = None
        # self.camera_subscription = self.create_subscription(
        #     Image, 
        #     self.topicsToSubscribe['image_rgb'], 
        #     self.callback, 
        #     qos_profile=self.qosProfile
        # )
        super().initRosComm(callbackObject=self)
    
    def peopleIntroducingCB(self, req: PeopleIntroducing.Request, res: PeopleIntroducing.Response):
        self.get_logger().info(f"Received introduce person request: {req.name}")
        
        name = req.name
        num_images = req.num_images
        
        previous_detection = None
        face_embeddings = []
        for idx in range(num_images):
            self.get_logger().info(f'Taking picture {idx+1} of {num_images} for {name}...')
            self.regressiveCounter(req.interval)
            num_retries = 3
            retry_count = 0
            while self.last_detection == previous_detection and retry_count < num_retries:
                self.get_logger().info("Waiting for new face detection...")
                self.regressiveCounter(1)
                retry_count += 1
            if retry_count >= num_retries:
                self.get_logger().warning("No new face detection found, skipping this image.")
                continue

            previous_detection = self.last_detection

            face_detection = None
            for detection in self.last_detection.detections:
                if detection.label == 'unknown':
                    face_detection = detection
                    break
            if not face_detection:
                self.get_logger().warn("No unknown face detection found, skipping this image.")
                continue
            
            face_embedding = face_detection.embedding
            if not face_embedding:
                self.get_logger().warn("No face embedding found, skipping this image.")
                continue
            face_embeddings.append(face_embedding)
            self.get_logger().info(f"Appending face embedding {idx} for {name}.")
        
        try:
            res.uuid = self.storeFaceEmbeddings(name, face_embeddings)
            res.response = True
            self.get_logger().info(f"Face embedding for {name} with stored successfully.")
        except ValueError as e:
            self.get_logger().error(f"Error storing face embeddings for {name}: {e}")
            res.response = False
            res.uuid = ''
    
        return res

    def callback(self, imageMsg: Image):
        
        self.get_logger().info("Face recognition callback triggered")

        face_recognitions = Detection3DArray()
        face_recognitions.header = imageMsg.header
        face_recognitions.image_rgb = copy.deepcopy(imageMsg) 
        cv_image = self.cvBridge.imgmsg_to_cv2(imageMsg, desired_encoding="bgr8")
        debug_image = self.cvBridge.imgmsg_to_cv2(imageMsg) 

        try:
            face_detections = DeepFace.represent(
                img_path=cv_image,
                model_name=self.deepface_model_name,
                detector_backend= self.deepface_detector_backend,
            )
        except ValueError as e:
            face_detections = []

        names = []
        if len(face_detections) > 0:
            nearest_neighbours = self.searchKNN([detection['embedding'] for detection in face_detections])

        for idx, face_detection in enumerate(face_detections):
            top, left = face_detection['facial_area']['y'], face_detection['facial_area']['x']
            right = left + face_detection['facial_area']['w']
            bottom = top + face_detection['facial_area']['h']
            confidence = face_detection['face_confidence']

            detection = Detection3D()
            if not nearest_neighbours[idx]:
                name = 'unknown'
                uuid = ''
            else:
                name = nearest_neighbours[idx]['name']
                uuid = nearest_neighbours[idx]['uuid']

            detection.label = name
            detection.uuid = uuid
            detection.embedding = face_detection['embedding']
            names.append(name)

            detectionHeader = imageMsg.header
            detection.header = detectionHeader
            size = int(right-left), int(bottom-top)
            
            detection.bbox2d.center.position.x = float(int(left) + int(size[1]/2))
            detection.bbox2d.center.position.y = float(int(top) + int(size[0]/2))
            detection.bbox2d.size_x = float(bottom-top)
            detection.bbox2d.size_y = float(right-left)

            # bb2d = BoundingBox2D()
            # data = BoundingBoxProcessingData()
            # data.sensor.setSensorData(cameraInfoMsg, depthMsg)


            # data.boundingBox2D.center.position.x = float(int(left) + int(size[1]/2))
            # data.boundingBox2D.center.position.y = float(int(top) + int(size[0]/2))
            # data.boundingBox2D.size_x = float(bottom-top)
            # data.boundingBox2D.size_y = float(right-left)
            # data.maxSize.x = float(3)
            # data.maxSize.y = float(3)
            # data.maxSize.z = float(3)

            # bb2d = data.boundingBox2D
    
            # try:
            #     bb3d = boundingBoxProcessing(data)
            # except Exception as e:
            #     self.get_logger().error(f"Error processing bounding box: {e}")
            #     continue

            cv2.rectangle(debug_image, (left, top), (right, bottom), (0, 255, 0), 2)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(debug_image, name, (left + 4, bottom - 4), font, 0.5, (0,0,255), 2)
            detection3d = detection #self.createDetection3d(bb2d, bb3d, detectionHeader, name)
            if detection3d is not None:
                face_recognitions.detections.append(detection3d)

        self.debugPublisher.publish(self.cvBridge.cv2_to_imgmsg(np.array(debug_image), encoding='rgb8'))

        if len(face_recognitions.detections) > 0:
            self.faceRecognitionPublisher.publish(face_recognitions)
            self.last_detection = face_recognitions


    def configureRedis(self):
        self.get_logger().info("Configuring Redis for face recognition")
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.index_name = "face_recognition_index"

        self.deepface_model_name = 'Facenet512'
        self.embedding_dim = 512
        self.deepface_detector_backend = 'yolov8'
        self.distance_metric = 'COSINE'  # 'L2', 'IP' OR 'COSINE'

        # Define the schema for the index
        schema = (
            TextField("uuid", no_stem=True),
            TextField("name", no_stem=True),
            VectorField(
                "embedding", 
                "FLAT", 
                {
                    "TYPE": "FLOAT32", 
                    "DIM": self.embedding_dim, 
                    "DISTANCE_METRIC": self.distance_metric
                    }
            ),
        )
        definition = IndexDefinition(prefix=["faces:"])
        # Check if the index already exists
        try:
            self.redis_client.ft(self.index_name).info()
            self.get_logger().info("Redis index already exists.")
            # Verify if the existing index matches the defined schema
            index_info = self.redis_client.ft(self.index_name).info()
            self.get_logger().info(f"Current index info: {index_info['attributes']}")

            for attribute in index_info["attributes"]:
                if 'embedding' in attribute and 'VECTOR' in attribute:
                    if (not self.embedding_dim in attribute) or (not self.distance_metric in attribute): 
                        self.get_logger().info("Redis index schema mismatch, deleting and recreating the index.")
                        self.redis_client.ft(self.index_name).dropindex(delete_documents=True)
                        self.redis_client.ft(self.index_name).create_index(fields=schema, definition=definition)
                    break
        except redis.exceptions.ResponseError:
            self.get_logger().info("Redis index does not exist, creating it.")
            self.redis_client.ft(self.index_name).create_index(fields=schema, definition=definition)
        
        info = self.redis_client.ft(self.index_name).info()
        self.get_logger().info(f"Redis index created. It has {info['num_docs']} documents.")
        # Delete existing documents in the Redis index
        # self.get_logger().info("Deleting existing documents in Redis index.")
        
        
        # try:
        #     keys = self.redis_client.keys("faces:*")
        #     if keys:
        #         self.redis_client.delete(*keys)
        #         self.get_logger().info(f"Deleted {len(keys)} existing documents from Redis index.")
        #     else:
        #         self.get_logger().info("No existing documents found in Redis index.")
        # except Exception as e:
        #     self.get_logger().error(f"Error while deleting existing documents: {e}")

    def storeFaceEmbeddings(self, name: str, embeddings: list, ttl=60*10):
        uuid = str(uuid4())
        self.get_logger().info(f"Storing face embedding for {name} with UUID {uuid}.")
        
        if not embeddings or len(embeddings) == 0:
            self.get_logger().warning(f"No embeddings provided for {name}. Skipping storage.")
            raise ValueError("No embeddings provided for storage.")
            
        
        for idx, embedding in enumerate(embeddings):
            if len(embedding) != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
            
            face_data = {
                "uuid": uuid,
                "name": name,
                "embedding": np.array(embedding, dtype=np.float32).tobytes()  # Convert embedding to bytes
            }
            redis_key = f"faces:{uuid}:{idx}"
            num_fields = self.redis_client.hset(redis_key, mapping=face_data)
            self.redis_client.expire(redis_key, ttl)
            self.get_logger().info(f"Stored face info with {num_fields} fields for {name} with UUID {uuid}:{idx} in Redis.")
        
        return uuid
    
    def searchKNN(self, embeddings, k=1):
        self.get_logger().info(f"Searching for {k} nearest neighbours for the provided embeddings.")
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        if not embeddings:
            return None
        # Convert embeddings to a list of lists
        
        query = (
            Query(f"*=>[KNN {k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("uuid", "name", "score")#.paging(0, k).bf("1.0")
            .dialect(2)
        )
        results_list = []
        for i, embedding in enumerate(embeddings):
            if len(embedding) != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
            result = (
                self.redis_client.ft(self.index_name).search(
                    query,
                    query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
                )
            ).docs
            embedding_result = []
            for doc in result:
                score = round(1 - float(doc.score), 2)
                self.get_logger().info(f"Found {doc.name} with UUID {doc.uuid} and score {score}")
                if score < self.threshold:
                    self.get_logger().info(f"Score {score} is below threshold {self.threshold}, skipping {doc.name}.")
                    continue
                embedding_result.append({
                    "uuid": doc.uuid,
                    "name": doc.name,
                    "score": score
                })
            results_list.append(embedding_result)

        #TODO: CHOOSE CHEAPER COMBINATION OF THE NEIGHBOURS FOUND

        first_results = [embedding_result[0] if embedding_result else None for embedding_result in results_list]
        return first_results


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
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
