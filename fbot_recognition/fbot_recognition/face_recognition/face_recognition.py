import rclpy

from fbot_recognition import BaseRecognition

from sensor_msgs.msg import Image, CameraInfo


class FaceRecognition(BaseRecognition):
    def __init__(self):
        super().__init__(packageName='fbot_recognition', nodeName='face_recognition')
        self.declareParameters()
        self.readParameters()
        self.loadModel()
        self.initRosComm()

    def initRosComm(self):
        super().initRosComm(callbackObject=self)

    def loadModel(self):
        pass

    def unLoadModel(self):
        pass

    def callback(self, depthMsg: Image, imageMsg: Image, cameraInfoMsg: CameraInfo):
        pass

    def declareParameters(self):
        super().declareParameters()
        self.declare_parameter('model_path', 'weights/face_recognition/face_recognition.pth')

    def readParameters(self):
        super().readParameters()
        self.modelPath = self.pkgPath + '/' + self.get_parameter('model_path').value


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()