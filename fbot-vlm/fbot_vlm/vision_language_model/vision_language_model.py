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


class VisionLanguageModel(Node):
    def __init__(self):
        super().__init__('vision_language_model')
        
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

        self.load_params(filename='vision_language_model.yaml')
        self.read_parameters()

        if self.vlm_api_type == 'ollama':
            self.vlm = ChatOllama(model=self.vlm_api_model, base_url=self.vlm_api_host)
        elif self.vlm_api_type == 'openai':
            self.vlm = ChatOpenAI(model_name=self.vlm_api_model, openai_api_base=self.vlm_api_host)
        elif self.vlm_api_type == 'google-genai':
            self.vlm = ChatGoogleGenerativeAI(model=self.vlm_api_model, convert_system_message_to_human=True)
        else:
            raise ValueError(f"VLM API type must be one of: {['ollama', 'openai', 'google-genai']}!")

        self.get_logger().info(f"VisionLanguageModel initialized with API type: {self.vlm_api_type}, model: {self.vlm_api_model}, host: {self.vlm_api_host}")

        self.answer_history: dict[str, list[VLMAnswer]] = {}
        self.vlm_service = self.create_service(VLMQuestionAnswering, self.vlm_service_name, self.handleServiceQuestionAnswering)
        self.vlm_question_subscriber = self.create_subscription(VLMQuestion, self.vlm_question_topic, self.handleTopicQuestion, 10)
        qos_profile = QoSProfile(depth=1)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.vlm_answer_publisher = self.create_publisher(VLMAnswer, self.vlm_answer_topic, qos_profile)
        self.vlm_answer_history_service = self.create_service(VLMAnswerHistory, self.vlm_history_service_name, self.handleAnswerHistoryService)


    def handleServiceQuestionAnswering(self, req: VLMQuestionAnswering.Request, res: VLMQuestionAnswering.Response):
        self.get_logger().info(f"Received question: {req.question}, use_image: {req.use_image}")
        try:
            self.validateHostConnection()
            self.validateQuestion(req)
            message = self.getHumanMessage(req.question, use_image=req.use_image, image=req.image)
            result = self.vlm.invoke([message,])
            res.success = True
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
            self.get_logger().info(f"Answering question: {req.question} with answer: {res.answer}")
            return res
    
    def handleAnswerHistoryService(self, req: VLMAnswerHistory.Request, res: VLMAnswerHistory.Response):
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
        for answers in self.answer_history.values():
            self.get_logger().info(f"Found {len(answers)} answers for question in history.")
        res.total_history_length = sum(len(answers) for answers in self.answer_history.values())
        self.get_logger().info(f"Returning {len(res.answers)} answers from history.")
        return res
        
    def handleTopicQuestion(self, msg: VLMQuestion):
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

        answer_msg.question = msg.question
        answer_msg.stamp = rclpy.time.Time().to_msg()
        self.vlm_answer_publisher.publish(answer_msg)
        self.storeAnswer(answer_msg)

    def storeAnswer(self, answer_obj: VLMAnswer):
        if answer_obj.question not in self.answer_history:
            self.answer_history[answer_obj.question] = []
            answer_obj.question_id = 0
        else:
            answer_obj.question_id = len(self.answer_history[answer_obj.question])
        self.answer_history[answer_obj.question].append(answer_obj)

    
    def validateHostConnection(self, timeout=5):
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
    
    def validateQuestion(self, msg: VLMQuestion | VLMAnswerHistory.Request):
        """
        Valida os parâmetros necessários para o serviço de VLM.
        """
        if not msg.question or msg.question == '' or msg.question == 'None':
            raise ValueError("Question cannot be empty.")
        if msg.use_image and (not msg.image or msg.image == Image()):
            self.get_logger().warn("Image is not provided, but use_image is set to True. Waiting for an image on the topic instead.")
        if msg.use_image and not isinstance(msg.image, Image):
            raise ValueError("Image must be of type sensor_msgs/Image or None when use_image is True.")
        
    def getHumanMessage(self, question: str, use_image: bool = False, image: Image = None):
        """
        Cria uma mensagem de humano com a pergunta e a imagem.
        """
        try:

            if use_image:
                return HumanMessage(
                    content=[
                        self.get_image_content(image),
                        {
                            'type': 'text',
                            'text': question
                        }
                    ]
                )
            else:
                return HumanMessage(
                    content=[
                        {
                            'type': 'text',
                            'text': question
                        }
                    ]
                )
        except Exception as e:
            raise ValueError(f"Failed to create human message: {e}")
        
    def get_image_content(self, question_image: Image = None):
        # Converte a mensagem de imagem para OpenCV
        if not question_image or not isinstance(question_image, Image) or question_image == Image():
            self.get_logger().info(f"Waiting for image on topic {self.rgb_image_topic} with timeout {self.rgb_image_timeout} seconds.")
            success, self.rgb_image_msg = wait_for_message(Image, self, self.rgb_image_topic, qos_profile=10, time_to_wait=self.rgb_image_timeout)
            
            if not success:
                raise ValueError(f"Timeout waiting for image on topic {self.rgb_image_topic} after {self.rgb_image_timeout} seconds.")
        else:
            self.get_logger().info(f"Using provided image for question: {question_image}")
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

    def read_parameters(self):
        package_share_dir = get_package_share_directory('fbot_vlm')
        dotenv_path = os.path.join(package_share_dir, '.env')
        load_dotenv(dotenv_path=dotenv_path)
        self.vlm_api_type = self.get_parameter('vlm_api_type').value
        self.vlm_api_host = self.get_parameter('vlm_api_host').value
        self.vlm_api_model = self.get_parameter('vlm_api_model').value
        self.rgb_image_topic = self.get_parameter('subscribers/image_rgb/topic').value
        self.rgb_image_timeout = self.get_parameter('subscribers/image_rgb/timeout').value
        self.vlm_service_name = self.get_parameter('servers/question_answering/service').value
        self.vlm_question_topic = self.get_parameter('subscribers/question/topic').value
        self.vlm_answer_topic = self.get_parameter('publishers/answer/topic').value
        self.vlm_history_service_name = self.get_parameter('servers/answer_history/service').value
    
    def load_params(self, filename):
        try:
            with open(os.path.join(get_package_share_directory('fbot_vlm'), 'config', filename)) as config_file:
                config = yaml.safe_load(config_file)[self.get_name()]['ros__parameters']
        except FileNotFoundError:
            self.get_logger().error(f"Configuration file {filename} not found.")
            raise
        except KeyError as e:
            self.get_logger().error(f"Missing key in configuration file: {e}")
            raise

        self.declare_parameters_from_dict(config)
            
    def declare_parameters_from_dict(self, params, path=''):
        for key, value in params.items():
            if isinstance(value, dict):
                self.declare_parameters_from_dict(value, path + key + '/')
            else:
                self.declare_parameter(path + key, value)

def main(args=None):
    rclpy.init(args=args)
    node = VisionLanguageModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()