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

#TODO: Change Detection3DArray message type for face recognition to FaceDetection3DArray for better semantic typing
class FaceRecognition(BaseRecognition):
    """!
    @brief ROS2 node for face detection and recognition system that combines YOLOv8 for face detection and DeepFace (Facenet512) for face recognition. 
    It uses Redis as a vector database to store and search face embeddings, providing real-time face recognition capabilities.
    """
    
    def __init__(self):
        """!
        @brief Initializes the face recognition system by setting up Redis connection, loading YOLO model, configuring parameters, and establishing ROS communication.
        """
        super().__init__(packageName='fbot_recognition', nodeName='face_recognition')
        packagePath = get_package_share_directory('fbot_recognition')
        self.declareParameters()
        self.readParameters()
        
        self.configureRedis()

        self.yolo_model = YOLO(os.path.join(packagePath, 'weights/yolov8l_100e.pt'))
        time.sleep(2)
        self.initRosComm()

    def initRosComm(self):
        """!
        @brief Initialize ROS2 communication channels
        """
        self.debugPublisher = self.create_publisher(Image, self.debugImageTopic, qos_profile=self.debugQosProfile)
        self.faceRecognitionPublisher = self.create_publisher(Detection3DArray, self.faceRecognitionTopic,  qos_profile=self.faceRecognitionQosProfile)
        service_cb_group = ReentrantCallbackGroup()
        self.introducePersonService = self.create_service(srv_type=PeopleIntroducing, srv_name=self.introducePersonServername, callback=self.peopleIntroducingCB, callback_group=service_cb_group)
        self.last_detection = None
        self.peopleForgettingService = self.create_service(srv_type=PeopleForgetting, srv_name=self.forgetPersonServername, callback=self.forgetPersonCB, callback_group=service_cb_group)
        super().initRosComm(callbackObject=self)
    
    def forgetPersonCB(self, req: PeopleForgetting.Request, res: PeopleForgetting.Response):
        """!
        @brief Service callback for forgetting a person, removing face embeddings from Redis database for a specified person
        @param req Service request containing person name, UUID, and protected entries
        @param res Service response indicating success and number of deleted entries
        @return PeopleForgetting.Response with success status and deletion count
        """
        num_deleted = self.deleteFaceEmbeddings(req.name, req.uuid, req.protected_names, req.protected_uuids)
        res.success = num_deleted > 0
        res.num_deleted = num_deleted
        return res
    
    def peopleIntroducingCB(self, req: PeopleIntroducing.Request, res: PeopleIntroducing.Response):
        """!
        @brief Service callback for introducing a new person. It captures multiple face images of a person, extracts embeddings, and stores them in Redis
        @param req Service request containing person name, number of images, and capture interval
        @param res Service response with success status and generated UUID
        @return PeopleIntroducing.Response with operation result and person UUID
        """
        name = req.name
        num_images = req.num_images
        
        previous_detection = None
        face_embeddings = []
        for idx in range(num_images):
            self.regressiveCounter(req.interval)
            num_retries = 3
            retry_count = 0
            while self.last_detection == previous_detection and retry_count < num_retries:
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
        
        try:
            res.uuid = self.storeFaceEmbeddings(name, face_embeddings)
            res.response = True
        except ValueError as e:
            self.get_logger().error(f"Error storing face embeddings for {name}: {e}")
            res.response = False
            res.uuid = ''
    
        return res

    def callback(self, depthMsg: Image, imageMsg: Image, cameraInfoMsg: CameraInfo):
        """!
        @brief Main callback for synchronized camera data processing. It recognizes known people, calculates 3D positions, and publishes results with debug visualization
        @param depthMsg ROS Image message containing depth information
        @param imageMsg ROS Image message containing RGB camera data
        @param cameraInfoMsg Camera calibration parameters
        """
        face_recognitions = Detection3DArray()
        face_recognitions.header = imageMsg.header
        face_recognitions.image_rgb = copy.deepcopy(imageMsg) 

        cv_image = self.cvBridge.imgmsg_to_cv2(imageMsg)
        debug_image = copy.deepcopy(cv_image)

        face_detections = self.detectFacesInImage(cv_image)

        self.unknown_idx = 0

        if len(face_detections) > 0:
            nearest_neighbours = self.searchKNN([detection['embedding'] for detection in face_detections], len(face_detections))

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

            if np.linalg.norm([bbox3d.center.position.x, bbox3d.center.position.y, bbox3d.center.position.z]) < 0.05:
                self.get_logger().warn(f"Face detection too close to the camera, skipping detection.")
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

    def orderByCloseness(self, detections):
        """!
        @brief Sort detections by distance from camera.
        @param detections List of Detection3D objects to be sorted
        @return List of Detection3D objects sorted by distance (closest first)
        """
        if len(detections) == 0:
            return []

        detections_with_dist = []
        for detection in detections:
            pose = detection.bbox3d.center.position
            dist = np.linalg.norm([pose.x, pose.y, pose.z])

            if dist <= self.faces_max_distance:
                detections_with_dist.append((detection, dist))
            else:
                continue

        detections_sorted = sorted(detections_with_dist, key=lambda x: x[1])
        detections_sorted = [detection[0] for detection in detections_sorted]
        return detections_sorted
                
    def detectFacesInImage(self, cv_image):
        """!
        @brief Detect faces in image and extract embeddings, using YOLOv8 and DeepFace.
        @param cv_image OpenCV image in BGR format
        @return List of dictionaries containing facial area, confidence, and embedding
        """
        detections=[]
        try:
            results = self.yolo_model.track(cv_image, classes=0, conf=self.threshold, imgsz=480)
            boxes = results[0].boxes
            
            for idx, box in enumerate(boxes):
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
        """!
        @brief Create 2D and 3D bounding boxes for detected face
        @param cameraInfoMsg Camera calibration parameters
        @param depthMsg Depth image message
        @param top Top coordinate of face bounding box
        @param right Right coordinate of face bounding box  
        @param left Left coordinate of face bounding box
        @param bottom Bottom coordinate of face bounding box
        @return Tuple containing BoundingBox2D and BoundingBox3D objects
        """

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
        """!
        @brief Create a Detection3D message for recognized face
        @param bb2d 2D bounding box of the detected face
        @param bb3d 3D bounding box of the detected face
        @param detectionHeader ROS message header with timestamp and frame
        @param label Person name or "unknown" if not recognized
        @param uuid Unique identifier for the person
        @param embedding Face embedding vector from DeepFace
        @return Detection3D message containing all face detection information
        """
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
        """!
        @brief Configure Redis vector database for face embeddings
        """
        self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
        self.get_logger().info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
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
            index_info = self.redis_client.ft(self.index_name).info()

            for attribute in index_info["attributes"]:
                if 'embedding' in attribute and 'VECTOR' in attribute:
                    if (not self.embedding_dim in attribute) or (not self.distance_metric in attribute): 
                        self.redis_client.ft(self.index_name).dropindex(delete_documents=True)
                        self.redis_client.ft(self.index_name).create_index(fields=schema, definition=definition)
                    break
        except redis.exceptions.ResponseError:
            self.redis_client.ft(self.index_name).create_index(fields=schema, definition=definition)
        
    def deleteFaceEmbeddings(self, name: str = None, uuid: str = None, protected_names: list = None, protected_uuids: list = None) -> int:
        """!
        @brief Removes stored face embeddings based on name and/or UUID, with protection for specified entries
        @param name Name of person whose embeddings should be deleted (optional)
        @param uuid UUID of person whose embeddings should be deleted (optional)
        @param protected_names List of names that should not be deleted (optional)
        @param protected_uuids List of UUIDs that should not be deleted (optional)
        @return Number of embeddings successfully deleted
        """
        if name and name!='' and uuid and uuid!='':
            # Delete embeddings for the specific name and UUID
            keys = self.redis_client.keys(f"faces:{name}:{uuid}:*")
            
        elif name and name!='':
            # Delete all embeddings for the given name
            keys = self.redis_client.keys(f"faces:{name}:*")
        elif uuid and uuid!='':
            # Delete all embeddings for the given UUID
            keys = self.redis_client.keys(f"faces:*:{uuid}:*")
        else:
            # Delete all embeddings
            keys = self.redis_client.keys("faces:*")
            
        if protected_names and protected_names != []:
            keys = [key for key in keys if not any(protected_name in key for protected_name in protected_names)]
        if protected_uuids and protected_uuids != []:
            keys = [key for key in keys if not any(protected_uuid in key for protected_uuid in protected_uuids)]

        add_message = f" for {name}" if name else ""
        add_message += f" with UUID {uuid}" if uuid else ""

        if keys:
            self.redis_client.delete(*keys)
            num_deleted = len(keys)
        else:
            num_deleted = 0

        return num_deleted

    def storeFaceEmbeddings(self, name: str, embeddings: list, ttl=60*10*6):
        """!
        @brief Store face embeddings in Redis database for a person with generated UUID and expiration time
        @param name Name of the person to store embeddings for
        @param embeddings List of face embedding vectors to store
        @param ttl Time-to-live in seconds for stored embeddings (default: 1 hour)
        @return Generated UUID for the stored embeddings
        """
        uuid = str(uuid4())
        
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
        
        return uuid
    
    def searchKNN(self, embeddings, k=1):
        """!
        @brief Search for K-nearest neighbors in face embedding space
        @param embeddings List of face embedding vectors to search for
        @param k Number of nearest neighbors to search for each embedding
        @return List of best matches with name, UUID, and similarity score for each input embedding
        """
        k=k*6
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
                if score < self.knn_threshhold:
                    continue

                if doc.uuid not in uuid_to_best_match or score > uuid_to_best_match[doc.uuid]["score"]:
                    uuid_to_best_match[doc.uuid] = {
                        "uuid": doc.uuid,
                        "name": doc.name,
                        "score": score,
                    }
            
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
        """!
        @brief Declare ROS2 node parameters
        """
        self.declare_parameter("publishers.debug.qos_profile", 1)
        self.declare_parameter("publishers.debug.topic", "/fbot_vision/fr/debug_face")
        self.declare_parameter("publishers.face_recognition.qos_profile", 1)
        self.declare_parameter("publishers.face_recognition.topic", "/fbot_vision/fr/face_recognition")
        self.declare_parameter("servers.introduce_person.servername", "/fbot_vision/fr/introduce_person")
        self.declare_parameter('model_path', 'weights/face_recognition/face_recognition.pth')
        self.declare_parameter("threshold", 0.8)
        self.declare_parameter("knn_threshold", 0.5)
        self.declare_parameter("redis_host", "localhost")
        self.declare_parameter("redis_port", 6379)
        self.declare_parameter("servers.forget_person.servername", "/fbot_vision/fr/forget_person")
        super().declareParameters()

    def readParameters(self):
        """!
        @brief Read and store ROS2 node parameters
        """
        self.debugImageTopic = self.get_parameter("publishers.debug.topic").value
        self.debugQosProfile = self.get_parameter("publishers.debug.qos_profile").value
        self.faceRecognitionTopic = self.get_parameter("publishers.face_recognition.topic").value
        self.faceRecognitionQosProfile = self.get_parameter("publishers.face_recognition.qos_profile").value
        self.introducePersonServername = self.get_parameter("servers.introduce_person.servername").value
        self.forgetPersonServername = self.get_parameter("servers.forget_person.servername").value
        self.threshold = self.get_parameter("threshold").value
        self.knn_threshhold = self.get_parameter("knn_threshold").value
        self.redis_host = self.get_parameter("redis_host").value
        self.redis_port = self.get_parameter("redis_port").value

        self.deepface_model_name = 'Facenet512'
        self.embedding_dim = 512
        self.deepface_detector_backend = 'yolov8'
        self.distance_metric = 'COSINE'  # 'L2', 'IP' OR 'COSINE'
        self.faces_max_distance = 4 #meters

        super().readParameters()
        self.modelPath = self.pkgPath + '/' + self.get_parameter('model_path').value

    def regressiveCounter(self, sec):
        """!
        @brief Display countdown timer for user feedback, useful during face capture sequences
        @param sec Number of seconds to count down from
        """
        sec = int(sec)
        for i  in range(0, sec):
            self.get_logger().warning(str(sec-i) + '...')
            time.sleep(1)

def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognition()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
