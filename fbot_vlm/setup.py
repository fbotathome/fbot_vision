from setuptools import find_packages, setup
import os
import glob

package_name = 'fbot_vlm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob.glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob.glob('config/.env')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='richassis',
    maintainer_email='richardjcassis@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_language_model = fbot_vlm.vision_language_model.vision_language_model:main',
        ],
    },
)
