#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import rclpy.time
from rclpy.wait_for_message import wait_for_message

from sensor_msgs.msg import Image
from fbot_vision_msgs.srv import VLMQuestionAnswering, VLMAnswerHistory
from fbot_vision_msgs.msg import VLMQuestion, VLMAnswer
from PIL import Image as IMG
from cv_bridge import CvBridge
import os
import yaml
from ament_index_python.packages import get_package_share_directory

import base64
from io import BytesIO
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
import requests
from requests.exceptions import RequestException

try:
    from langchain_ollama import ChatOllama
except:
    pass
try:
    from langchain_openai.chat_models import ChatOpenAI
except:
    pass
try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
except:
    pass
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup



class VisionLanguageModel(Node):
    def __init__(self, ):
        """
        @brief A Node that provides Vision Language Model (VLM) capabilities in a flexible way, allowing for different VLM APIs and models,
        and handling image input for question answering. 
        """
        super().__init__('vision_language_model', )
        
        self.bridge = CvBridge()
        self.vlm_api_type = None
        self.vlm_api_host = None
        self.vlm_api_model = None
        self.rgb_image_topic = None
        self.rgb_image_timeout = None
        self.vlm_service_name = None
        self.vlm_question_topic = None
        self.vlm_answer_topic = None
        self.vlm_history_service_name = None
        self.loadParams(filename='vision_language_model.yaml')
        self.readParameters()

        self.answer_history: dict[str, list[VLMAnswer]] = {}
        self.initClientVLM()
        self.initRosComm()

    def initClientVLM(self) -> None:
        """
        @brief Initializes the VLM client based on the API type and model specified in the parameters.
        """
        if self.vlm_api_type == 'ollama':
            self.vlm = ChatOllama(model=self.vlm_api_model, base_url=self.vlm_api_host)
        elif self.vlm_api_type == 'openai':
            self.vlm = ChatOpenAI(model_name=self.vlm_api_model, openai_api_base=self.vlm_api_host)
        elif self.vlm_api_type == 'google-genai':
            self.vlm = ChatGoogleGenerativeAI(model=self.vlm_api_model)
        else:
            raise ValueError(f"VLM API type must be one of: {['ollama', 'openai', 'google-genai']}!")

        self.get_logger().info(f"VisionLanguageModel initialized with API type: {self.vlm_api_type}, model: {self.vlm_api_model}, host: {self.vlm_api_host}")

    def initRosComm(self) -> None:
        """
        @brief Initializes the ROS communication components, including service and topic subscriptions.
        """
        service_cb_group = ReentrantCallbackGroup()
        self.vlm_service = self.create_service(VLMQuestionAnswering, self.vlm_service_name, self.handleServiceQuestionAnswering, callback_group=service_cb_group)
        topic_cb_group = ReentrantCallbackGroup()
        self.vlm_question_subscriber = self.create_subscription(VLMQuestion, self.vlm_question_topic, self.handleTopicQuestion, 10, callback_group=topic_cb_group)
        qos_profile = QoSProfile(depth=1)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.vlm_answer_publisher = self.create_publisher(VLMAnswer, self.vlm_answer_topic, qos_profile)
        self.vlm_answer_history_service = self.create_service(VLMAnswerHistory, self.vlm_history_service_name, self.handleAnswerHistoryService)


    def handleServiceQuestionAnswering(self, req: VLMQuestionAnswering.Request, res: VLMQuestionAnswering.Response) -> VLMQuestionAnswering.Response:
        """
        @brief Callback function for the VLM question answering service.
        @param req: The request containing the question and optional image.
        @param res: The response containing the answer, success status, confidence, and timestamp.
        """
        try:
            self.validateHostConnection()
            self.validateQuestion(req)
            message = self.getHumanMessage(req.question, use_image=req.use_image, image=req.image)
            result = self.vlm.invoke([message,])
            res.success = True
            self.get_logger().info(f"VLM invoked successfully with question: {req.question}")
            res.answer = result.content
            res.confidence = 1.0
        except Exception as e:
            self.get_logger().error(f"Error processing question: {e}")
            res.success = False
            res.answer = "ERROR: " + str(e)
            res.confidence = 0.0
        finally:
            res.stamp = rclpy.time.Time().to_msg()
            res.question = req.question
            self.storeAnswer(VLMAnswer(
                question=req.question,
                answer=res.answer,
                success=res.success,
                confidence=res.confidence,
                stamp=res.stamp
            ))
            self.get_logger().info(f"VLM answered with: {res.answer}")
            return res
    
    def handleAnswerHistoryService(self, req: VLMAnswerHistory.Request, res: VLMAnswerHistory.Response) -> VLMAnswerHistory.Response:
        """ 
        @brief Callback function for the VLM answer history service, providing a list with the history of answers.
        @param req: The request containing the filter for questions.
        @param res: The response containing the list of answers and the total history length.
        """
        res.answers = []
        if not req.questions_filter is None and len(req.questions_filter) > 0:
            for question in req.questions_filter:
                if question in self.answer_history:
                    answers = self.answer_history[question]
                    res.answers.extend(answers)
                else:
                    self.get_logger().warn(f"Question '{question}' not found in history.")
        else:
            # Return all answers in the history
            for answers in self.answer_history.values():
                res.answers.extend(answers)
        res.total_history_length = sum(len(answers) for answers in self.answer_history.values())
        self.get_logger().info(f"Returning {len(res.answers)} answers from history.")
        return res
        
    def handleTopicQuestion(self, msg: VLMQuestion) -> None:
        """
        @brief Callback function for the VLM question topic subscriber, processing incoming questions and invoking the VLM.
        @param msg: The incoming VLMQuestion message containing the question and optional image.
        """
        self.get_logger().info(f"Received question: {msg.question} with use_image={msg.use_image}")
        answer_msg = VLMAnswer()

        try:
            self.validateHostConnection()
            self.validateQuestion(msg)
            message = self.getHumanMessage(msg.question, use_image=msg.use_image, image=msg.image)
            result = self.vlm.invoke([message,])
            answer_msg.answer = result.content
            answer_msg.success = True
            answer_msg.confidence = 1.0
        except Exception as e:
            self.get_logger().error(f"Error processing question: {e}")
            answer_msg.answer = "ERROR: " + str(e)
            answer_msg.success = False
            answer_msg.confidence = 0.0

        self.get_logger().info(f"VLM answered with: {answer_msg.answer}")
        answer_msg.question = msg.question
        answer_msg.stamp = rclpy.time.Time().to_msg()
        self.vlm_answer_publisher.publish(answer_msg)
        self.storeAnswer(answer_msg)

    def storeAnswer(self, answer_obj: VLMAnswer) -> None:
        """
        @brief Stores the answer in the answer history, creating a new entry if the question is not already present.
        @param answer_obj: The VLMAnswer object containing the question, answer, success status, confidence, and timestamp.
        """
        if answer_obj.question not in self.answer_history:
            self.answer_history[answer_obj.question] = []
            answer_obj.question_id = 0
        else:
            answer_obj.question_id = len(self.answer_history[answer_obj.question])
        self.answer_history[answer_obj.question].append(answer_obj)

    
    def validateHostConnection(self, timeout=5) -> bool:
        """
        @brief Validates the connection to the VLM API host.
        @param timeout: The timeout in seconds for the connection attempt.
        @return: True if the connection is successful, raises ConnectionError otherwise.
        """
        try:
            response = requests.get(self.vlm_api_host, timeout=timeout)
            if response.status_code == 200:
                self.get_logger().info(f"Conexão com o host {self.vlm_api_host} validada com sucesso.")
                return True
            else:
                error_message = f"Host {self.vlm_api_host} respondeu com status {response.status_code}."
                raise ConnectionError(error_message)
        except RequestException as e:
            error_message = f"Erro ao conectar ao host {self.vlm_api_host}: {e}"
            raise ConnectionError(error_message)
    
    def validateQuestion(self, msg: VLMQuestion | VLMAnswerHistory.Request) -> None:
        """
        @brief Validates the question in the incoming message.
        @param msg: The incoming message containing the question and optional image.
        @raises ValueError: If the question is empty or if the image is not of the correct type.
        """
        if not msg.question or msg.question == '' or msg.question == 'None':
            raise ValueError("Question cannot be empty.")
        if msg.use_image and (not msg.image or msg.image == Image()):
            self.get_logger().warn("Image is not provided, but use_image is set to True. Waiting for an image on the topic instead.")
        if msg.use_image and not isinstance(msg.image, Image):
            raise ValueError("Image must be of type sensor_msgs/Image or None when use_image is True.")
        
    def getHumanMessage(self, question: str, use_image: bool = False, image: Image = None) -> HumanMessage:
        """
        @brief Creates a HumanMessage for the VLM, optionally including an image.
        @param question: The question to be asked.
        @param use_image: Whether to include an image in the message.
        @param image: The image to be included in the message, if use_image is True
        @return: A HumanMessage object containing the question and image.
        """
        try:
            if use_image:
                return HumanMessage(content=[self.getImageContent(image),{'type': 'text','text': question}])
            else:
                return HumanMessage(
                    content=[{'type': 'text','text': question}])
        except Exception as e:
            raise ValueError(f"Failed to create human message: {e}")
        
    def getImageContent(self, question_image: Image = None) -> dict:
        """
        @brief Retrieves the image content for the question, either from the provided image or by waiting for an image on the topic.
        @param question_image: The image to be used for the question, if provided.
        @return: A dictionary containing the image URL in base64 format.
        """

        if not question_image or not isinstance(question_image, Image) or question_image == Image():
            self.get_logger().info(f"Waiting for image on topic {self.rgb_image_topic} with timeout {self.rgb_image_timeout} seconds.")
            success, self.rgb_image_msg = wait_for_message(Image, self, self.rgb_image_topic, qos_profile=10, time_to_wait=self.rgb_image_timeout)
            
            if not success:
                raise ValueError(f"Timeout waiting for image on topic {self.rgb_image_topic} after {self.rgb_image_timeout} seconds.")
        else:
            self.get_logger().info(f"Using provided image for question.")
            self.rgb_image_msg = question_image
        
        buffered = BytesIO()
        cv_image = self.bridge.imgmsg_to_cv2(self.rgb_image_msg, desired_encoding='rgb8')
        img = IMG.fromarray(cv_image)  # Usa PIL.Image corretamente
        img.save(buffered, format='JPEG')
        b64_image_str = base64.b64encode(buffered.getvalue()).decode()
        if self.vlm_api_type in ('ollama',):
            return {
                'type': 'image_url',
                'image_url': f"data:image/jpeg;base64,{b64_image_str}"
            }
        else:
            return {
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/jpeg;base64,{b64_image_str}"
                }
            }

    def readParameters(self) -> None:
        """
        @brief Reads parameters from the node and loads environment variables if necessary.
        """
        package_share_dir = get_package_share_directory('fbot_vlm')
        dotenv_path = os.path.join(package_share_dir, 'config', '.env')
        self.vlm_api_type = self.get_parameter('vlm_api_type').value
        self.vlm_api_host = self.get_parameter('vlm_api_host').value
        self.vlm_api_model = self.get_parameter('vlm_api_model').value
        self.rgb_image_topic = self.get_parameter('subscribers/image_rgb/topic').value
        self.rgb_image_timeout = self.get_parameter('subscribers/image_rgb/timeout').value
        self.vlm_service_name = self.get_parameter('servers/question_answering/service').value
        self.vlm_question_topic = self.get_parameter('subscribers/question/topic').value
        self.vlm_answer_topic = self.get_parameter('publishers/answer/topic').value
        self.vlm_history_service_name = self.get_parameter('servers/answer_history/service').value
        load_dotenv(dotenv_path=dotenv_path)
    
    def loadParams(self, filename) -> None:
        """
        @brief Loads parameters from a YAML configuration file.
        @param filename: The name of the YAML file containing the parameters.
        """
        try:
            with open(os.path.join(get_package_share_directory('fbot_vlm'), 'config', filename)) as config_file:
                config = yaml.safe_load(config_file)[self.get_name()]['ros__parameters']
        except FileNotFoundError:
            self.get_logger().error(f"Configuration file {filename} not found.")
            raise
        except KeyError as e:
            self.get_logger().error(f"Missing key in configuration file: {e}")
            raise

        self.declareParametersFromDict(config)
        self.declare_parameter('vlm_api_type', 'ollama')
        self.declare_parameter('vlm_api_host', 'http://localhost:11434')
        self.declare_parameter('vlm_api_model', 'gemma3:4b')
            
    def declareParametersFromDict(self, params, path='') -> None:
        """
        @brief Recursively declares parameters from a dictionary.
        @param params: The dictionary containing parameters to declare.
        @param path: The path prefix for the parameters.
        """
        for key, value in params.items():
            if isinstance(value, dict):
                self.declareParametersFromDict(value, path + key + '/')
            else:
                self.declare_parameter(path + key, value)

def main(args=None):
    rclpy.init(args=args)
    node = VisionLanguageModel()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()