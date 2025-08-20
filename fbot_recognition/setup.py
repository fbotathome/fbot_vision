from setuptools import find_packages, setup
import os
import glob

package_name = 'fbot_recognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob.glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config/yolo_tracker_config'), glob.glob('config/yolo_tracker_config/*.yaml')),
        (os.path.join('share', package_name, 'weights'), glob.glob('weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gdorneles',
    maintainer_email='dorneles1215@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov8_recognition = fbot_recognition.yolov8_recognition.yolov8_recognition:main',
            'yolo_tracker_recognition = fbot_recognition.yolo_tracker_recognition.yolo_tracker_recognition:main',
            'face_recognition = fbot_recognition.face_recognition.face_recognition:main',
            'moondream_recognition = fbot_recognition.moondream_recognition.moondream_recognition:main',
        ],
    },
)
