#!/usr/bin/env python3
from math import e
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from fbot_vision_msgs.srv import VisualQuestionAnswering 
import PIL

from cv_bridge import CvBridge
import os
import yaml
from ament_index_python.packages import get_package_share_directory

import base64
from io import BytesIO
from langchain_core.messages import HumanMessage

try:
    from langchain_community.chat_models.ollama import ChatOllama
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


class VisionLanguageModel(Node):
    def __init__(self):
        super().__init__('vision_language_model')
        
        self.bridge = CvBridge()

        self.load_params(filename='vision_language_model.yaml')
        self.read_parameters()

        if self.vlm_api_type == 'ollama':
            self.vlm = ChatOllama(model=self.vlm_api_model, base_url=self.vlm_api_host)
        elif self.vlm_api_type == 'openai':
            self.vlm = ChatOpenAI(model_name=self.vlm_api_model, openai_api_base=self.vlm_api_host, api_key=self.vlm_api_key)
        elif self.vlm_api_type == 'google-genai':
            self.vlm = ChatGoogleGenerativeAI(model=self.vlm_api_model, convert_system_message_to_human=True)
        else:
            raise ValueError(f"VLM API type must be one of: {['ollama', 'openai', 'google-genai']}!")
        print('vlm = ', self.vlm)
        self.image_rgb_subscriber = self.create_subscription(Image, self.rgb_image_topic, self._update_rgb_image, 10)
        self.visual_question_answering_server = self.create_service(VisualQuestionAnswering, self.visual_question_answering_service, self._handle_visual_question_answering)

    def _update_rgb_image(self, msg: Image):
        self.rgb_image_msg = msg

    def _handle_visual_question_answering(self, req: VisualQuestionAnswering.Request):
        message = HumanMessage(
            content=[
                self.get_image_content(),
                {
                    'type': 'text',
                    'text': f'{req.question}'
                }
            ]
        )
        res = VisualQuestionAnswering.Response()
        res.answer = self.vlm.invoke([message,]).content
        res.confidence = 1.0
        return res

    def get_image_content(self):
        # rospy.wait_for_message(self.rgb_image_topic, Image)
        # Reavaliar se precisa esperar mensagem, ou se posso considerar que a imagem já chegou
        buffered = BytesIO()
        cv_image = self.bridge.imgmsg_to_cv2(self.rgb_image_msg, desired_encoding='rgb8')
        img = PIL.Image.fromarray(cv_image)
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
        self.vlm_api_type = self.get_parameter('vlm_api_type').value
        self.vlm_api_host = self.get_parameter('vlm_api_host').value
        self.vlm_api_model = self.get_parameter('vlm_api_model').value
        self.vlm_api_key = self.get_parameter('vlm_api_key').value
        self.rgb_image_topic = self.get_parameter('subscribers/image_rgb/topic').value
        self.visual_question_answering_service = self.get_parameter('servers/visual_question_answering/service').value
    
    def load_params(self, filename):
        with open(os.path.join(get_package_share_directory('fbot_vision_language_model'), 'config', filename)) as config_file:
            config = yaml.safe_load(config_file)[self.get_name()]['ros__parameters']

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