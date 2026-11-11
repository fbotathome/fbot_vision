#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import copy
import time
import json
import base64
import numpy as np
import cv2
import rclpy

from openai import OpenAI
from PIL import Image as IMG
from image2world.image2worldlib import *
from fbot_recognition import BaseRecognition

from std_msgs.msg import Header, String
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from fbot_vision_msgs.msg import Detection3D, Detection3DArray
from vision_msgs.msg import BoundingBox2D, BoundingBox3D

OBJECTS_PROMPTS = {
    # Class: fabrics
    "white_shirt":      "white t-shirt",
    "grey_shirt":        "grey t-shirt",
    "blue_shirt":         "blue t-shirt",
    "black_shirt":       "black t-shirt",
    "hand_towel":        "folded hand towel",

    # Class: toys
    "rubiks_cube":       "rubik's cube",

    # Class: snacks
    "pringles":          "pringles can",
    "seaweed":            "seaweed snack pack",

    # Class: fruits
    "apple":               "apple",
    "peach":               "peach",
    "mangostane":       "mangosteen fruit",
    "lemon":               "lemon",
    "yellow_bellpepper": "yellow bell pepper",
    "red_bellpepper":    "red bell pepper",

    # Class: food
    "instant_noodles": "instant noodles bag",
    "cornflakes":        "cornflakes cup",

    # Class: drinks
    "coke":                 "coca-cola can",
    "red_bull":           "red bull can",
    "milk":                 "milk carton",
    "soju":                 "soju bottle",
    "pepsi":               "pepsi bottle",

    # Class: cleaning_supplies
    "dishwasher_tab":  "dishwasher tablet",
    "sponge":              "yellow sponge",
    "toothpaste":        "colgate toothpaste box",

    # Class: dishes
    "cup":                   "red mug",
    "spoon":               "spoon",
    "plate":                "red plate",
    "knife":                "knife",
    "fork":                  "fork",
    "bowl":                 "red bowl",
}

from ament_index_python.packages import get_package_share_directory
class GeminiRecognition(BaseRecognition):
    def __init__(self) -> None:
        super().__init__(nodeName='gemini_recognition')

        self.labels_dict: dict = {}
        self.targetPhrases: list = []
        self.phraseToKey: dict = {}
        self.declareParameters()
        self.readParameters()
        self.loadModel()
        self.initRosComm()

    def initRosComm(self) -> None:
        self.debugPublisher = self.create_publisher(Image, self.debugImageTopic, qos_profile=self.debugQosProfile)
        self.markerPublisher = self.create_publisher(MarkerArray, 'fbot_vision/fr/gemini_markers', qos_profile=self.debugQosProfile)
        self.objectRecognitionPublisher = self.create_publisher(Detection3DArray, self.objectRecognitionTopic, qos_profile=self.objectRecognitionQosProfile)
        self.objectPromptSubscriber = self.create_subscription(String, self.objectPromptTopic, qos_profile=self.qosProfile, callback=self.updateObjectPrompt)
        super().initRosComm(callbackObject=self)

    def loadModel(self) -> None:
        self.get_logger().info("=> Setting up OpenRouter client")
        apiKey = self.apiKey or os.environ.get("OPENROUTER_API_KEY")
        if not apiKey:
            self.get_logger().error(
                "No OpenRouter API key found. Set the 'model.api_key' parameter "
                "or the OPENROUTER_API_KEY environment variable.")
            raise RuntimeError("Missing OpenRouter API key")
        self.client = OpenAI(base_url=self.baseUrl, api_key=apiKey)
        self.get_logger().info(f"=> Ready (model: {self.modelName} via {self.baseUrl})")

    def unLoadModel(self) -> None:
        self.client = None

    def updateObjectPrompt(self, msg: String):
        # Accept a single class or a comma-separated list of classes.
        tokens = [t.strip() for t in msg.data.split(',') if t.strip()]
        self.targetPhrases = []
        self.phraseToKey = {}
        for key in tokens:
            phrase = OBJECTS_PROMPTS.get(key, key)
            self.targetPhrases.append(phrase)
            # The model may answer with either the human phrase or the raw key;
            # map both back to the canonical class key.
            self.phraseToKey[phrase.lower()] = key
            self.phraseToKey[key.lower()] = key
        if self.targetPhrases:
            self.get_logger().info(f"Object prompts updated to: {self.targetPhrases}")
        else:
            self.get_logger().info("Object prompts cleared")

    def detect(self, pilImage: IMG.Image, labels: list) -> list:
        """Query the Gemini model through OpenRouter for every requested label in
        a single request and return a list of boxes. Each box carries normalized
        coordinates in the same format as the moondream node
        ({x_min, y_min, x_max, y_max} in the [0, 1] range) plus a canonical
        "label" resolved back to the requested class key."""
        labelList = ", ".join(f"'{label}'" for label in labels)
        prompt = (
            f"Detect all instances of the following objects in the image: {labelList}. "
            "Respond with a JSON object with a single key \"detections\" whose value "
            "is an array. Each array element is an object with two keys: \"label\", "
            "set to exactly one of the requested object names, and \"box_2d\", the "
            "bounding box as [ymin, xmin, ymax, xmax] normalized to 0-1000. Include "
            "one element per detected instance. If there are no instances, return "
            "{\"detections\": []}. Return only the JSON, with no extra text."
        )

        dataUrl = self.encodeImage(pilImage)

        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": dataUrl}},
                    ],
                }],
            )
        except Exception as e:
            self.get_logger().error(f"OpenRouter API request failed: {e}")
            return []

        content = response.choices[0].message.content if response.choices else None
        if not content:
            self.get_logger().error("Empty response from OpenRouter")
            return []

        boxes = []
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            self.get_logger().error(f"Failed to parse OpenRouter response: {e}")
            return []

        # Accept either {"detections": [...]} or a bare list.
        if isinstance(data, dict):
            data = data.get("detections", [])

        for item in data:
            if not isinstance(item, dict):
                continue
            box = item.get("box_2d")
            if not box or len(box) != 4:
                continue
            rawLabel = str(item.get("label", "")).strip()
            key = self.phraseToKey.get(rawLabel.lower(), rawLabel)
            y_min, x_min, y_max, x_max = box
            boxes.append({
                "x_min": min(x_min, x_max) / 1000.0,
                "y_min": min(y_min, y_max) / 1000.0,
                "x_max": max(x_min, x_max) / 1000.0,
                "y_max": max(y_min, y_max) / 1000.0,
                "label": key,
            })
        return boxes

    @staticmethod
    def encodeImage(pilImage: IMG.Image) -> str:
        """Encode a PIL image as a base64 JPEG data URL."""
        bgr = np.array(pilImage)[..., ::-1]
        ok, buffer = cv2.imencode('.jpg', bgr)
        if not ok:
            raise RuntimeError("Failed to JPEG-encode image")
        b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    def callback(self, depthMsg: Image, imageMsg: Image, cameraInfoMsg: CameraInfo) -> None:

        if not self.targetPhrases:
            if time.time() % 5 < 0.1:
                self.get_logger().warn("Waiting for object prompt to be set ...")
            return

        if imageMsg is None or depthMsg is None or cameraInfoMsg is None:
            self.get_logger().error("One or more input messages are invalid.")
            return

        cvImage = self.cvBridge.imgmsg_to_cv2(imageMsg, desired_encoding='bgr8')
        pilImage = IMG.fromarray(cvImage[..., ::-1])
        results = self.detect(pilImage, self.targetPhrases)

        detectionHeader = imageMsg.header

        detection3DArray = Detection3DArray()
        detection3DArray.header = detectionHeader
        detection3DArray.image_rgb = imageMsg

        if len(results):
            for box in results:

                score = 1.0

                bb2d = BoundingBox2D()
                data = BoundingBoxProcessingData()
                data.sensor.setSensorData(cameraInfoMsg, depthMsg)

                x_min = int(box['x_min'] * pilImage.width)
                x_max = int(box['x_max'] * pilImage.width)

                y_min = int(box['y_min'] * pilImage.height)
                y_max = int(box['y_max'] * pilImage.height)

                centerX = float((x_max + x_min)/2.0)
                centerY = float((y_max + y_min)/2.0)

                sizeX = float(x_max - x_min)
                sizeY = float(y_max - y_min)

                data.boundingBox2D.center.position.x = centerX
                data.boundingBox2D.center.position.y = centerY
                data.boundingBox2D.size_x = sizeX
                data.boundingBox2D.size_y = sizeY
                data.maxSize.x = self.maxSizes[0]
                data.maxSize.y = self.maxSizes[1]
                data.maxSize.z = self.maxSizes[2]

                bb2d = data.boundingBox2D

                try:
                    bb3d = boundingBoxProcessing(data)
                except Exception as e:
                    self.get_logger().error(f"Error processing bounding box: {e}")
                    continue

                detection3d = self.createDetection3d(bb2d, bb3d, score, detectionHeader, box['label'])
                if detection3d is not None:
                    detection3DArray.detections.append(detection3d)

        self.objectRecognitionPublisher.publish(detection3DArray)
        self.labels_dict.clear()

        imageArray = cvImage.copy()
        if len(results):
            for box in results:
                x_min = int(box['x_min'] * pilImage.width)
                x_max = int(box['x_max'] * pilImage.width)

                y_min = int(box['y_min'] * pilImage.height)
                y_max = int(box['y_max'] * pilImage.height)

                imageArray = cv2.rectangle(imageArray, (x_min, y_min), (x_max, y_max), (255, 0, 255), 6)

                label = box['label']

                box_w = max(1, x_max - x_min)
                box_h = max(1, y_max - y_min)
                font_scale = max(1.0, min(4.5, min(box_w, box_h) / 135.0))
                thickness = max(1, int(round(font_scale * 2)))

                (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                pad_x = 6
                pad_y = 4
                text_x = x_min
                text_y = max(y_min - pad_y, h + pad_y)
                top_left = (text_x - pad_x, text_y - h - pad_y)
                bottom_right = (text_x + w + pad_x, text_y + baseline + pad_y)

                cv2.rectangle(imageArray, top_left, bottom_right, (255, 0, 255), -1)
                cv2.putText(imageArray, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        image = IMG.fromarray(imageArray[..., ::-1])
        debugImageMsg = self.cvBridge.cv2_to_imgmsg(np.array(image), encoding='rgb8')
        self.debugPublisher.publish(debugImageMsg)

        self.publishMarkers(detection3DArray.detections)

    def createDetection3d(self, bb2d: BoundingBox2D, bb3d: BoundingBox3D , score: float, detectionHeader: Header, label: str) -> Detection3D:
        detection3d = Detection3D()
        detection3d.header = detectionHeader
        detection3d.score = score

        if not label:
            label = "object"

        if '/' in label:
            detection3d.label = label
        else:
            detection3d.label = f"none-{label}" if label[0].islower() else f"None-{label}"

        if detection3d.label in self.labels_dict:
            self.labels_dict[detection3d.label] += 1
        else:
            self.labels_dict[detection3d.label] = 1

        detection3d.id = self.labels_dict[detection3d.label]

        detection3d.bbox2d = copy.deepcopy(bb2d)
        detection3d.bbox3d = bb3d

        return detection3d


    def publishMarkers(self, descriptions3d) -> None:
        markers = MarkerArray()
        duration = Duration()
        duration.sec = 2
        color = np.asarray([255, 0, 0])/255.0
        for i, det in enumerate(descriptions3d):
            name = det.label

            # cube marker
            marker = Marker()
            marker.header = det.header
            marker.action = Marker.ADD
            marker.pose = det.bbox3d.center
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.4
            marker.ns = "bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.scale = det.bbox3d.size
            marker.lifetime = duration
            markers.markers.append(marker)

            # text marker
            marker = Marker()
            marker.header = det.header
            marker.action = Marker.ADD
            marker.pose = det.bbox3d.center
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker.id = i
            marker.ns = "texts"
            marker.type = Marker.TEXT_VIEW_FACING
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.text = '{} ({:.2f})'.format(name, det.score)
            marker.lifetime = duration
            markers.markers.append(marker)

        self.markerPublisher.publish(markers)

    def declareParameters(self) -> None:
        self.declare_parameter("publishers.debug.topic", "/fbot_vision/fr/debug")
        self.declare_parameter("publishers.debug.qos_profile", 1)
        self.declare_parameter("publishers.object_recognition.topic", "/fbot_vision/fr/object_recognition")
        self.declare_parameter("publishers.object_recognition.qos_profile", 1)
        self.declare_parameter("max_sizes", [0.05, 0.05, 0.05])
        self.declare_parameter("subscribers.object_prompt", "/fbot_vision/fr/object_prompt")
        self.declare_parameter("model.name", "google/gemini-2.5-flash")
        self.declare_parameter("model.api_key", "")
        self.declare_parameter("model.base_url", "https://openrouter.ai/api/v1")
        super().declareParameters()

    def readParameters(self) -> None:
        self.debugImageTopic = self.get_parameter("publishers.debug.topic").value
        self.debugQosProfile = self.get_parameter("publishers.debug.qos_profile").value
        self.objectRecognitionTopic = self.get_parameter("publishers.object_recognition.topic").value
        self.objectRecognitionQosProfile = self.get_parameter("publishers.object_recognition.qos_profile").value
        self.maxSizes = self.get_parameter("max_sizes").value
        self.objectPromptTopic = self.get_parameter("subscribers.object_prompt").value
        self.modelName = self.get_parameter("model.name").value
        self.apiKey = self.get_parameter("model.api_key").value
        self.baseUrl = self.get_parameter("model.base_url").value
        super().readParameters()

def main(args=None):
    rclpy.init(args=args)
    node = GeminiRecognition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
