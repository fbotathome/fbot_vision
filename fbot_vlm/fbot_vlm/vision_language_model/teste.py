# test_cvbridge_conversion.py
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Caminho da imagem
image_path = '/home/marina/Documentos/FBot/pessoa1.jpg'

# Carrega a imagem com OpenCV
cv_image = cv2.imread(image_path)
if cv_image is None:
    print("Erro ao carregar a imagem!")
    exit(1)

# Converte para sensor_msgs.msg.Image
bridge = CvBridge()
ros_image = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

# Verifica alguns campos da mensagem convertida
print("Conversão realizada com sucesso!")
print(f"Altura: {ros_image.height}")
print(f"Largura: {ros_image.width}")
print(f"Encoding: {ros_image.encoding}")
print(f"Tamanho do data: {len(ros_image.data)} bytes")