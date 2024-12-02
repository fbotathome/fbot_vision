import rclpy

from fbot_recognition import BaseRecognition


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

    def callback(self, *args):
        pass

    def declareParameters(self):
        super().declareParameters()
        self.declare_parameter('model_path', 'weights/face_recognition/face_recognition.pth')

    def readParameters(self):
        super().readParameters()
        self.modelPath = self.pkgPath + '/' + self.get_parameter('model_path').value