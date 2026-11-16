#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import copy
import time
import json
import base64
import numpy as np
import re
import cv2
import threading
import requests
import rclpy

from openai import OpenAI
from rclpy.executors import ExternalShutdownException
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

# Local models often wrap the JSON in a markdown code fence even with response_format=json.
CODE_FENCE = re.compile(r'^\s*```[a-zA-Z]*\s*\n?(.*?)\n?\s*```\s*$', re.DOTALL)

from ament_index_python.packages import get_package_share_directory
class GeminiRecognition(BaseRecognition):
    def __init__(self) -> None:
        super().__init__(nodeName='gemini_recognition')

        self.labels_dict: dict = {}
        self.targetPhrases: list = []
        self.phraseToKey: dict = {}
        self.declareParameters()
        self.readParameters()
        self.primaryDownUntil = 0.0
        self.loadModel()

        # The OpenRouter request takes seconds; it runs in a worker thread so the
        # executor keeps serving the camera and prompt subscriptions meanwhile.
        self.frameLock = threading.Lock()
        self.latestFrame = None
        self.frameReady = threading.Event()
        self.stopWorker = threading.Event()
        self.worker = threading.Thread(target=self.detectionLoop, daemon=True)

        self.initRosComm()
        self.worker.start()

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
        # Without a timeout the client waits up to 600 s and retries twice.
        self.client = OpenAI(base_url=self.baseUrl, api_key=apiKey,
                             timeout=self.requestTimeout, max_retries=0)
        self.get_logger().info(
            f"=> Ready (model: {self.modelName} via {self.baseUrl}, timeout {self.requestTimeout} s, "
            f"max_tokens {self.maxTokens}, max_image_size {self.maxImageSize})")

        self.fallbackClient = None
        if self.fallbackModel:
            # Ollama's OpenAI-compatible API (on the Jetson) answers when OpenRouter is unreachable.
            self.fallbackClient = OpenAI(base_url=f"{self.fallbackHost.rstrip('/')}/v1", api_key="ollama",
                                         timeout=self.fallbackTimeout, max_retries=0)
            self.get_logger().info(f"=> Fallback: {self.fallbackModel} via {self.fallbackHost} (timeout {self.fallbackTimeout} s)")
            if self.fallbackKeepLoaded:
                threading.Thread(target=self.preloadFallback, daemon=True).start()
        else:
            self.get_logger().info("=> Fallback: disabled")

    def preloadFallback(self) -> None:
        """Loads the fallback model in Ollama and pins it in memory (keep_alive=-1), so the
        first fallback request does not also pay the model load."""
        try:
            response = requests.post(f"{self.fallbackHost.rstrip('/')}/api/generate",
                                     json={"model": self.fallbackModel, "keep_alive": -1}, timeout=120)
            response.raise_for_status()
            self.get_logger().info(f"Fallback model {self.fallbackModel} loaded on {self.fallbackHost}")
        except Exception as e:
            self.get_logger().warn(f"Could not preload fallback model {self.fallbackModel}: {e}")

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

        dataUrl = self.encodeImage(pilImage, self.maxImageSize)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": dataUrl}},
            ],
        }]

        content = None
        primaryFailed = False
        if time.time() >= self.primaryDownUntil:
            extraBody = {"reasoning": {"enabled": False}} if self.disableReasoning else None
            content, primaryFailed = self.requestDetections(self.client, self.modelName, "OpenRouter", messages, extraBody)
            if primaryFailed:
                # Skip the primary for a while: otherwise every frame waits for another timeout.
                self.primaryDownUntil = time.time() + self.primaryRetryInterval
                if self.fallbackClient is not None:
                    self.get_logger().error(
                        f"Primary model failed; using fallback {self.fallbackModel} "
                        f"for the next {self.primaryRetryInterval:.0f} s")
        if content is None and self.fallbackClient is not None and (primaryFailed or time.time() < self.primaryDownUntil):
            # Disable "thinking" on reasoning-capable local models (gemma4): much faster.
            content, _ = self.requestDetections(self.fallbackClient, self.fallbackModel, "fallback",
                                                messages, {"reasoning_effort": "none"})
        if content is None:
            return []

        boxes = []
        match = CODE_FENCE.match(content)
        if match:
            content = match.group(1)
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            self.get_logger().error(f"Failed to parse model response: {e}")
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

    def requestDetections(self, client: OpenAI, model: str, name: str, messages: list, extraBody) -> tuple:
        """Sends one detection request. Returns (content or None, request_failed): request_failed
        is True when the endpoint did not answer (network, timeout, HTTP error), which is
        what triggers the fallback; an answer that is empty or cut by max_tokens is not."""
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                # Without a cap the model once looped for 4 min and returned 191 KB of JSON.
                max_tokens=self.maxTokens,
                extra_body=extraBody,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception as e:
            self.get_logger().error(f"{name} request ({model}) failed after {time.time() - start:.1f} s: {e}")
            return None, True

        elapsed = time.time() - start
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice else None
        if not content:
            self.get_logger().error(f"Empty response from {name} ({model}, {elapsed:.1f} s)")
            return None, False
        if choice.finish_reason == 'length':
            self.get_logger().error(f"{name} response ({model}) hit max_tokens ({self.maxTokens}) after {elapsed:.1f} s, discarding")
            return None, False
        log = self.get_logger().debug if client is self.client else self.get_logger().warn
        log(f"{name} ({model}) answered in {elapsed:.1f} s")
        return content, False

    @staticmethod
    def encodeImage(pilImage: IMG.Image, maxSize: int = 0) -> str:
        """Encode a PIL image as a base64 JPEG data URL, downscaled so its longer
        side is at most maxSize (0 keeps the original size). The model answers in
        coordinates normalized to the image, so scaling does not change the boxes."""
        if maxSize > 0 and max(pilImage.size) > maxSize:
            pilImage = pilImage.copy()
            pilImage.thumbnail((maxSize, maxSize))
        bgr = np.array(pilImage)[..., ::-1]
        ok, buffer = cv2.imencode('.jpg', bgr)
        if not ok:
            raise RuntimeError("Failed to JPEG-encode image")
        b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    def callback(self, depthMsg: Image, imageMsg: Image, cameraInfoMsg: CameraInfo) -> None:

        if not self.targetPhrases:
            # if time.time() % 5 < 0.1:
            #     self.get_logger().warn("Waiting for object prompt to be set ...")
                
            return

        if imageMsg is None or depthMsg is None or cameraInfoMsg is None:
            self.get_logger().error("One or more input messages are invalid.")
            return

        # Only keep the newest synchronized frame; detectionLoop consumes it.
        with self.frameLock:
            self.latestFrame = (depthMsg, imageMsg, cameraInfoMsg)
        self.frameReady.set()

    def detectionLoop(self) -> None:
        """Worker thread: one OpenRouter request at a time, on the newest frame,
        at most once every min_interval seconds."""
        lastRequest = 0.0
        while not self.stopWorker.is_set():
            if not self.frameReady.wait(timeout=0.5):
                continue
            wait = self.minInterval - (time.time() - lastRequest)
            if wait > 0 and self.stopWorker.wait(timeout=wait):
                break
            with self.frameLock:
                frame = self.latestFrame
                self.latestFrame = None
                self.frameReady.clear()
            if frame is None or not self.targetPhrases:
                continue
            lastRequest = time.time()
            try:
                self.processFrame(*frame)
            except Exception as e:
                self.get_logger().error(f"Gemini detection failed: {e}")

    def processFrame(self, depthMsg: Image, imageMsg: Image, cameraInfoMsg: CameraInfo) -> None:
        targetPhrases = list(self.targetPhrases)
        if not targetPhrases:
            return

        cvImage = self.cvBridge.imgmsg_to_cv2(imageMsg, desired_encoding='bgr8')
        pilImage = IMG.fromarray(cvImage[..., ::-1])
        results = self.detect(pilImage, targetPhrases)

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
                    # self.get_logger().error(f"Error processing bounding box: {e}")
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
        self.declare_parameter("model.timeout", 10.0)
        self.declare_parameter("model.max_tokens", 2048)
        self.declare_parameter("model.disable_reasoning", True)
        self.declare_parameter("model.max_image_size", 1024)
        self.declare_parameter("model.min_interval", 1.0)
        self.declare_parameter("model.primary_retry_interval", 30.0)
        self.declare_parameter("fallback.model", "gemma4:latest")
        self.declare_parameter("fallback.host", "http://jetson:11434")
        self.declare_parameter("fallback.timeout", 30.0)
        self.declare_parameter("fallback.keep_loaded", True)
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
        self.requestTimeout = float(self.get_parameter("model.timeout").value)
        self.maxTokens = int(self.get_parameter("model.max_tokens").value)
        self.disableReasoning = bool(self.get_parameter("model.disable_reasoning").value)
        self.maxImageSize = int(self.get_parameter("model.max_image_size").value)
        self.minInterval = float(self.get_parameter("model.min_interval").value)
        self.primaryRetryInterval = float(self.get_parameter("model.primary_retry_interval").value)
        self.fallbackModel = str(self.get_parameter("fallback.model").value).strip()
        if self.fallbackModel.lower() in ("", "none"):
            self.fallbackModel = ""
        self.fallbackHost = self.get_parameter("fallback.host").value
        self.fallbackTimeout = float(self.get_parameter("fallback.timeout").value)
        self.fallbackKeepLoaded = bool(self.get_parameter("fallback.keep_loaded").value)
        super().readParameters()

def main(args=None):
    rclpy.init(args=args)
    node = GeminiRecognition()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stopWorker.set()
        node.worker.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
