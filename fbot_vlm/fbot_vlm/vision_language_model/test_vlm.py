import cv2
import rclpy
from rclpy.node import Node
from fbot_vision_msgs.srv import VLMQuestionAnswering, VLMAnswerHistory
from fbot_vision_msgs.msg import VLMQuestion, VLMAnswer
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

class VisionLanguageModelTester(Node):
    def __init__(self):
        super().__init__('vision_language_model_tester')

        # Cliente para o serviço VLMQuestionAnswering
        self.vlm_service_client = self.create_client(VLMQuestionAnswering, '/fbot_vision/bvlm/question_answering/query')
        while not self.vlm_service_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Aguardando o serviço VLMQuestionAnswering estar disponível...')

        # Cliente para o serviço VLMAnswerHistory
        self.history_service_client = self.create_client(VLMAnswerHistory, '/fbot_vision/bvlm/answer_history/query')
        while not self.history_service_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Aguardando o serviço VLMAnswerHistory estar disponível...')

        # Publicador para o tópico VLMQuestion
        self.question_publisher = self.create_publisher(VLMQuestion, '/fbot_vision/bvlm/question_answering/question', 10)

        # Assinante para o tópico VLMAnswer
        self.topic_answer_received = False
        self.answer_subscriber = self.create_subscription(VLMAnswer, '/fbot_vision/bvlm/question_answering/answer', self.answer_callback, 10)
        # Testa os serviços e tópicos
        self.test_services_and_topics()

        origin_image = cv2.imread('/home/marina/Documentos/FBot/pessoa1.jpg')  
        bridge = CvBridge()
        self.image_msg = bridge.cv2_to_imgmsg(origin_image, encoding='bgr8')

    def test_services_and_topics(self):
        # Testa o serviço VLMQuestionAnswering
        self.get_logger().info('Chamando o serviço VLMQuestionAnswering...')
        request = VLMQuestionAnswering.Request()
        request.question = "Qual é o objeto na imagem?"
        request.use_image = False
        request.image = self.image_msg
        response = self.vlm_service_client.call(request)
        if response is None:
            self.get_logger().error('Falha ao chamar o serviço VLMQuestionAnswering')
        elif not response.success:
            self.get_logger().error('O serviço retornou falha ao processar a pergunta: {}'.format(response.answer))
        else:
            self.get_logger().info(f"Resposta do serviço VLMQuestionAnswering: {response.answer}, Confiança: {response.confidence}")
        
        # Publica uma mensagem no tópico VLMQuestion
        self.get_logger().info('Publicando no tópico VLMQuestion...')
        question_msg = VLMQuestion()
        question_msg.question = "Qual é o objeto na imagem?"
        question_msg.use_image = False
        self.question_publisher.publish(question_msg)

        # Aguarda um momento para garantir que a mensagem seja processada
        while not self.topic_answer_received:
            time.sleep(2)
        # Testa o serviço VLMAnswerHistory
        self.get_logger().info('Chamando o serviço VLMAnswerHistory...')
        history_request = VLMAnswerHistory.Request()
        history_request.questions_filter = ["Qual é o objeto na imagem?"]
        history_future = self.history_service_client.call_async(history_request)
        history_future.add_done_callback(self.history_response_callback)

    def service_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Resposta do serviço VLMQuestionAnswering: {response.answer}, Confiança: {response.confidence}")
        except Exception as e:
            self.get_logger().error(f"Erro ao chamar o serviço VLMQuestionAnswering: {e}")

    def history_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Histórico de respostas: {response.answers}")
        except Exception as e:
            self.get_logger().error(f"Erro ao chamar o serviço VLMAnswerHistory: {e}")

    def answer_callback(self, msg):
        self.get_logger().info(f"Resposta recebida no tópico VLMAnswer: {msg.answer}, Confiança: {msg.confidence}")
        self.topic_answer_received = True


def main(args=None):
    rclpy.init(args=args)
    node = VisionLanguageModelTester()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()