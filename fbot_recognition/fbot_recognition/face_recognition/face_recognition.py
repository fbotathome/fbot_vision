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
from fbot_vision_msgs.msg import Detection3D, FaceDetection3D, FaceDetection3DArray, Detection3DArray
from vision_msgs.msg import BoundingBox2D, BoundingBox3D
from fbot_vision_msgs.srv import PeopleIntroducing, PeopleForgetting
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
from ultralytics import YOLO


class FaceRecognition(BaseRecognition):
    def __init__(self):
        super().__init__(packageName='fbot_recognition', nodeName='face_recognition')
        packagePath = get_package_share_directory('fbot_recognition')
        datasetPath = os.path.join(packagePath, 'dataset')
        self.featuresPath = os.path.join(datasetPath, 'features')
        self.peopleDatasetPath = os.path.join(datasetPath, 'people/')
        self.declareParameters()
        self.readParameters()
        
        self.configureRedis()

        self.yolo_model = YOLO(os.path.join(packagePath, 'weights/yolov8l_100e.pt'))

        knownFacesDict = self.loadVar('features')
        self.knownFaces = self.flatten(knownFacesDict)
        time.sleep(2)
        self.initRosComm()

    def initRosComm(self):
        self.debugPublisher = self.create_publisher(Image, self.debugImageTopic, qos_profile=self.debugQosProfile)
        # self.faceRecognitionPublisher = self.create_publisher(FaceDetection3DArray, self.faceRecognitionTopic,  qos_profile=self.faceRecognitionQosProfile)
        self.faceRecognitionPublisher = self.create_publisher(Detection3DArray, self.faceRecognitionTopic,  qos_profile=self.faceRecognitionQosProfile)
        service_cb_group = ReentrantCallbackGroup()
        self.introducePersonService = self.create_service(srv_type=PeopleIntroducing, srv_name=self.introducePersonServername, callback=self.peopleIntroducingCB, callback_group=service_cb_group)
        self.last_detection = None
        self.peopleForgettingService = self.create_service(srv_type=PeopleForgetting, srv_name=self.forgetPersonServername, callback=self.forgetPersonCB, callback_group=service_cb_group)
        super().initRosComm(callbackObject=self)
    
    def forgetPersonCB(self, req: PeopleForgetting.Request, res: PeopleForgetting.Response):
        self.get_logger().info(f"Received forget person request: {req.name}")
        
        num_deleted = self.deleteFaceEmbeddings(req.name, req.uuid)

        res.success = num_deleted > 0
        res.num_deleted = num_deleted

        return res
    
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
                # if detection.detection.label == 'unknown':
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

    def callback(self, depthMsg: Image, imageMsg: Image, cameraInfoMsg: CameraInfo):
        
        self.get_logger().info("Face recognition callback triggered")

        # face_recognitions = FaceDetection3DArray()
        face_recognitions = Detection3DArray()
        face_recognitions.header = imageMsg.header
        face_recognitions.image_rgb = copy.deepcopy(imageMsg) 

        cv_image = self.cvBridge.imgmsg_to_cv2(imageMsg)
        debug_image = copy.deepcopy(cv_image)

        face_detections = self.detectFacesInImage(cv_image)

        self.unknown_idx = 0
        
        if len(face_detections) > 0:
            nearest_neighbours = self.searchKNN([detection['embedding'] for detection in face_detections], len(face_detections))
            self.get_logger().info(f"Found {len(nearest_neighbours)} nearest neighbours for detected faces.")

        for idx, face_detection in enumerate(face_detections):

            confidence = face_detection['face_confidence']

            top, left = face_detection['facial_area']['y'], face_detection['facial_area']['x']
            right = left + face_detection['facial_area']['w']
            bottom = top + face_detection['facial_area']['h']

            try:
                bbox2d, bbox3d = self.createBoundingBoxes(cameraInfoMsg, depthMsg, top, right, left, bottom)
            except Exception as e:
                self.get_logger().error(f"{e}. Skipping detection")
                continue

            if nearest_neighbours[idx]:
                name = nearest_neighbours[idx]['name']
                uuid = nearest_neighbours[idx]['uuid']
            else:
                name = f'unknown'
                uuid = f'{unknown_idx}'
                unknown_idx += 1
   
            cv2.rectangle(debug_image, (left, top), (right, bottom), (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(debug_image, name, (left + 4, bottom + 10), font, 0.5, (180,180,180), 2)


            faceDetection3D = self.createFaceDetection3D(bbox2d, bbox3d, imageMsg.header, name, uuid, face_detection['embedding'])
            face_recognitions.detections.append(faceDetection3D)

        self.debugPublisher.publish(self.cvBridge.cv2_to_imgmsg(np.array(debug_image), encoding='rgb8'))

        if len(face_recognitions.detections) > 0:

            face_recognitions.detections = self.orderByCloseness(face_recognitions.detections)

            self.faceRecognitionPublisher.publish(face_recognitions)
            self.last_detection = face_recognitions
            self.get_logger().warn(f'DETECTION PUBLISHED: {[detection.label for detection in face_recognitions.detections]}')

    def orderByCloseness(self, detections):
        if len(detections) == 0:
            return []

        detections_with_dist = []
        for detection in detections:
            pose = detection.bbox3d.center.position
            dist = np.linalg.norm([pose.x, pose.y, pose.z])
            detections_with_dist.append((detection, dist))

        detections_sorted = sorted(detections_with_dist, key=lambda x: x[1])
        detections_sorted = [detection[0] for detection in detections_sorted]
        return detections_sorted
                

    def detectFacesInImage(self, cv_image):
        detections=[]
        try:
            self.get_logger().info("Running YOLO model for face detection")
            results = self.yolo_model.track(cv_image, classes=0, conf=self.threshold, imgsz=480)
            boxes = results[0].boxes
            self.get_logger().info(f"Detected {len(boxes)} faces")  
            
            for idx, box in enumerate(boxes):
                self.get_logger().info(f"Processing box of index {idx}")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = cv_image[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                face_detection = DeepFace.represent(
                    img_path=face_crop,
                    model_name=self.deepface_model_name,
                    detector_backend= 'skip',
                )

                detections.append({
                    'facial_area': {
                        'x': x1,
                        'y': y1,
                        'w': x2 - x1,
                        'h': y2 - y1
                    },
                    'face_confidence': box.conf[0],
                    'embedding': face_detection[0]['embedding']
                })

        except Exception as e:
            self.get_logger().error(f'Error running YOLO for face detection: {e}')
        finally:
            return detections

    def createBoundingBoxes(self, cameraInfoMsg: CameraInfo, depthMsg: Image, top, right, left, bottom):

        size = int(right-left), int(bottom-top)

        data = BoundingBoxProcessingData()
        data.sensor.setSensorData(cameraInfoMsg, depthMsg)
        data.boundingBox2D.center.position.x = float(int(left) + int(size[1]/2))
        data.boundingBox2D.center.position.y = float(int(top) + int(size[0]/2))
        data.boundingBox2D.size_x = float(bottom-top)
        data.boundingBox2D.size_y = float(right-left)
        data.maxSize.x = float(3)
        data.maxSize.y = float(3)
        data.maxSize.z = float(3)

        try:
            bb3d = boundingBoxProcessing(data)
        except Exception as e:
            raise Exception("An error occurred while processing the bounding box.")
        
        return data.boundingBox2D, bb3d
    

    def createFaceDetection3D(self, bb2d: BoundingBox2D, bb3d: BoundingBox3D, detectionHeader: Header, label: str, uuid: str, embedding: list) -> FaceDetection3D:

        """USing FaceDetection3D message"""
        # faceDetection3D = FaceDetection3D()
        # faceDetection3D.detection.header = detectionHeader
        # faceDetection3D.detection.id = 0
        # faceDetection3D.detection.label = label
        # faceDetection3D.detection.bbox2d = copy.deepcopy(bb2d)
        # faceDetection3D.detection.bbox3d = copy.deepcopy(bb3d)

        """Using Detection3D message"""
        faceDetection3D = Detection3D()
        faceDetection3D.header = detectionHeader
        faceDetection3D.id = 0
        faceDetection3D.label = label
        faceDetection3D.bbox2d = copy.deepcopy(bb2d)
        faceDetection3D.bbox3d = copy.deepcopy(bb3d)
        faceDetection3D.uuid = uuid
        faceDetection3D.embedding = embedding

        return faceDetection3D

    def configureRedis(self):
        self.get_logger().info("Configuring Redis for face recognition")
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.index_name = "face_recognition_index"

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
        
        ####
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
        # ###
        
    def deleteFaceEmbeddings(self, name: str, uuid: str = None):
        """
        Delete face embeddings from Redis for a given name and optionally a UUID.
        
        :param name: The name associated with the embeddings to delete.
        :param uuid: (Optional) The UUID associated with the embeddings to delete.
        """
        if not name:
            raise ValueError("Name must be provided to delete embeddings.")
        
        if uuid is not None and uuid != '':
            # Delete embeddings for the specific name and UUID
            keys = self.redis_client.keys(f"faces:{name}:{uuid}:*")
            if keys:
                self.redis_client.delete(*keys)
                self.get_logger().info(f"Deleted {len(keys)} embeddings for {name} with UUID {uuid}.")
                num_deleted = len(keys)
            else:
                self.get_logger().info(f"No embeddings found for {name} with UUID {uuid}.")
                num_deleted = 0
        else:
            # Delete all embeddings for the given name
            keys = self.redis_client.keys(f"faces:{name}:*")
            if keys:
                self.redis_client.delete(*keys)
                self.get_logger().info(f"Deleted {len(keys)} embeddings for {name}.")
                num_deleted = len(keys)
            else:
                self.get_logger().info(f"No embeddings found for {name}.")
                num_deleted = 0

        return num_deleted

    def storeFaceEmbeddings(self, name: str, embeddings: list, ttl=60*10*6):
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
            redis_key = f"faces:{name}:{uuid}:{idx}"
            num_fields = self.redis_client.hset(redis_key, mapping=face_data)
            self.redis_client.expire(redis_key, ttl)
            self.get_logger().info(f"Stored face info with {num_fields} fields for {name} with UUID {uuid}:{idx} in Redis.")
        
        return uuid
    
    def searchKNN(self, embeddings, k=1):
        k=k*6
        # self.get_logger().info(f"Searching for {k} nearest neighbours for the provided embeddings.")
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        if not embeddings:
            return None

        query = (
            Query(f"*=>[KNN {k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("uuid", "name", "score", "embedding")
            .dialect(2)
        )
        results_list = []

        for i, embedding in enumerate(embeddings):
            self.get_logger().info(f"Processing embedding {i+1}/{len(embeddings)}.")
            if len(embedding) != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")

            result = (
                self.redis_client.ft(self.index_name).search(
                    query,
                    query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
                )
            ).docs

            # Group results by UUID and keep only the closest match for each UUID
            uuid_to_best_match = {}
            for doc in result:
                score = round(1 - float(doc.score), 2)
                self.get_logger().info(f"Found {doc.name} with UUID {doc.uuid} and score {score}")
                if score < self.knn_threshhold:
                    self.get_logger().info(f"Score {score} is below threshold {self.knn_threshhold}, skipping {doc.name}.")
                    continue

                if doc.uuid not in uuid_to_best_match or score > uuid_to_best_match[doc.uuid]["score"]:
                    uuid_to_best_match[doc.uuid] = {
                        "uuid": doc.uuid,
                        "name": doc.name,
                        "score": score,
                    }
            
            self.get_logger().info(f"Found {len(uuid_to_best_match)} unique matches for embedding {i+1}/{len(embeddings)}.")
            embedding_result = list(uuid_to_best_match.values())
            results_list.append(embedding_result)

        # Resolve conflicts where multiple embeddings match the same UUID
        assigned_uuids = set()
        final_results = [None] * len(embeddings)

        for i, embedding_result in enumerate(results_list):
            for match in embedding_result:
                if match["uuid"] not in assigned_uuids:
                    final_results[i] = match
                    assigned_uuids.add(match["uuid"])
                    break

        # Handle conflicts by assigning the highest score to one embedding and the next best match to the other
        for i, embedding_result in enumerate(results_list):
            if final_results[i] is None:  # If no match was assigned
                for match in embedding_result:
                    if match["uuid"] not in assigned_uuids:
                        final_results[i] = match
                        assigned_uuids.add(match["uuid"])
                        break
        
        for i in range(len(final_results)):
            if final_results[i] is None:
                final_results[i] = {
                    "uuid": f"{self.unknown_idx}",
                    "name": "unknown",
                    "score": 0.0,
                }
                self.unknown_idx += 1
        return final_results


    def declareParameters(self):
        self.declare_parameter("publishers.debug.qos_profile", 1)
        self.declare_parameter("publishers.debug.topic", "/fbot_vision/fr/debug_face")
        self.declare_parameter("publishers.face_recognition.qos_profile", 1)
        self.declare_parameter("publishers.face_recognition.topic", "/fbot_vision/fr/face_recognition")
        self.declare_parameter("servers.introduce_person.servername", "/fbot_vision/fr/introduce_person")
        self.declare_parameter('model_path', 'weights/face_recognition/face_recognition.pth')
        self.declare_parameter("threshold", 0.8)
        self.declare_parameter("knn_threshold", 0.5)
        self.declare_parameter("servers.forget_person.servername", "/fbot_vision/fr/forget_person")
        super().declareParameters()

    def readParameters(self):
        self.debugImageTopic = self.get_parameter("publishers.debug.topic").value
        self.debugQosProfile = self.get_parameter("publishers.debug.qos_profile").value
        self.faceRecognitionTopic = self.get_parameter("publishers.face_recognition.topic").value
        self.faceRecognitionQosProfile = self.get_parameter("publishers.face_recognition.qos_profile").value
        self.introducePersonServername = self.get_parameter("servers.introduce_person.servername").value
        self.forgetPersonServername = self.get_parameter("servers.forget_person.servername").value
        self.threshold = self.get_parameter("threshold").value
        self.knn_threshhold = self.get_parameter("knn_threshold").value

        self.deepface_model_name = 'Facenet512'
        self.embedding_dim = 512
        self.deepface_detector_backend = 'yolov8'
        self.distance_metric = 'COSINE'  # 'L2', 'IP' OR 'COSINE'

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



def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognition()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
