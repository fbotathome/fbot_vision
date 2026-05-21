#!/usr/bin/env python3
import math
import open3d as o3d
from typing import List
import numpy as np
import rclpy
import rclpy.logging
from rclpy.node import Node
from cv_bridge import CvBridge
import message_filters
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from fbot_vision_msgs.msg import Detection2DArray, Detection2D, Detection3DArray, Detection3D, KeyPoint3D, KeyPoint2D
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration
import cv2

import ros2_numpy

np.random.seed(72)

def quaternion_from_matrix(matrix):
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

def generateRandomColor():
    red = np.random.randint(0, 256)
    green = np.random.randint(0, 256)
    blue = np.random.randint(0, 256)
    return np.asarray((red, green, blue), dtype=np.uint8)

class Image2World(Node):

    def __init__(self):
        super().__init__('image_2_world')

        self.callbacks = {
            Detection2D.INSTANCE_SEGMENTATION : self.detectionSeg2D_to_detectionSeg3D,
            Detection2D.DETECTION : self.detection2D_to_detection3D,
            Detection2D.POSE : self.detectionPose2D_to_detectionPose3D
        }

        # Declare parameter
        # self.camera_name = self.get_parameter('camera_name').value

        self.cv_bridge = CvBridge()
        self.current_camera_info = None
        self.lut_table = None
        self.default_depth = 0.5
        self.label_to_color = {}

        # Publishers
        self._dbg_pub = self.create_publisher(Detection3DArray, f"/fbot_vision/i2w/detection3d", 1)
        self.pcd_publisher = self.create_publisher(PointCloud2, f"/fbot_vision/i2w/img_pcd", 1)
        self.marker_publisher = self.create_publisher(MarkerArray, "/fbot_vision/i2w/marker", 1)

        # Subscribers
        self.detection2d_sub = self.create_subscription(Detection2DArray, "/fbot_vision/fr/object_recognition",self.callback,10)

        
        self.get_logger().info(f"Node {self.get_name()} initiaded.")


    def __compareCameraInfo(self, camera_info: CameraInfo):
        equal = True
        equal = equal and (camera_info.width == self.current_camera_info.width)
        equal = equal and (camera_info.height == self.current_camera_info.height)
        equal = equal and np.all(np.isclose(np.asarray(camera_info.k),
                                            np.asarray(self.current_camera_info.k)))
        return equal
        
    def __mountLutTable(self, camera_info: CameraInfo):
        if self.lut_table is None or not self.__compareCameraInfo(camera_info):
            self.current_camera_info = camera_info
            K = np.asarray(camera_info.k).reshape((3,3))

            fx = 1./K[0,0]
            fy = 1./K[1,1]
            cx = K[0,2]
            cy = K[1,2]

            x_table = (np.arange(0, self.current_camera_info.width) - cx)*fx 
            y_table = (np.arange(0, self.current_camera_info.height) - cy)*fy

            x_mg, y_mg = np.meshgrid(x_table, y_table)

            self.lut_table = np.concatenate((x_mg[:, :, np.newaxis], y_mg[:, :, np.newaxis]), axis=2)

    def pointCloudArraystoOpen3D(self, xyz: np.ndarray):
        if len(xyz.shape) == 3:
            xyz = xyz.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.remove_non_finite_points()
        return pcd
    
    def Open3dToPointCloud2(self, pcd: o3d.geometry.PointCloud, header: Header):
        xyz = np.asarray(pcd.points)
        data: PointCloud2 = self.arrays2toPointCloud2(xyz, header)
        return data

    def detectionSeg2D_to_detectionSeg3D(self, detection2d: Detection2D, pcd: np.ndarray, header: Header) -> Detection3D: 
        # Convert the mask to a boolean array
        mask = self.cv_bridge.imgmsg_to_cv2(detection2d.mask, "passthrough") > 0

        # Resize the mask to match the height of the pcd
        mask_resized = cv2.resize(mask.astype(np.uint8), (pcd.shape[1], pcd.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        # points3d = pcd[mask]
        points3d = pcd[mask_resized]

        o3d_pcd = self.pointCloudArraystoOpen3D(points3d)
        o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=0.02)
        o3d_pcd, _ = o3d_pcd.remove_radius_outlier(nb_points=20,
                                                   radius=0.1)
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d_pcd.points)

        bbox_center = bbox.get_center()
        box_rotation = np.eye(4,4)
        #box_rotation[:3, :3] = box_r
        box_orientation = quaternion_from_matrix(box_rotation)
        bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
        bbox_size = np.dot(bbox_size, box_rotation[:3, :3])

        self.get_logger().info(f"{bbox_size}")

        # Create the 3D detection
        detection3d = Detection3D()
        detection3d.label = detection2d.label
        detection3d.class_num = detection2d.class_num
        detection3d.score = detection2d.score
        detection3d.bbox3d.center.position.x = float(bbox_center[0])
        detection3d.bbox3d.center.position.y = float(bbox_center[1])
        detection3d.bbox3d.center.position.z = float(bbox_center[2])
        detection3d.bbox3d.size.x = bbox_size[0]
        detection3d.bbox3d.size.y = bbox_size[1]
        detection3d.bbox3d.size.z = detection2d.max_size.z
        detection3d.bbox3d.center.orientation.x = float(box_orientation[0])
        detection3d.bbox3d.center.orientation.y = box_orientation[1]
        detection3d.bbox3d.center.orientation.z = box_orientation[2]
        detection3d.bbox3d.center.orientation.w = box_orientation[3]

        return detection3d
    
    def detectionPose2D_to_detectionPose3D(self, detection2d: Detection2D, pcd: np.ndarray, header : Header) -> Detection3D:
        detection3d = self.detection2D_to_detection3D(detection2d, pcd, header)
        kp : KeyPoint2D
        for kp in detection2d.pose:
            kp3d = KeyPoint3D()
            kp3d.id = kp.id
            kp3d.score = kp.score
            x3D, y3D, z3D = pcd[int(kp.y), int(kp.x)]
            kp3d.x = x3D
            kp3d.y = y3D
            kp3d.z = z3D
            detection3d.pose.append(kp3d)

        raise NotImplementedError("Method not implemented")
    
    def detection2D_to_detection3D(self, detection2d: Detection2D, pcd: np.ndarray, header : Header) -> Detection3D:
        
        x2D, y2D = int(detection2d.bbox.center.position.x), int(detection2d.bbox.center.position.y)
        center_points = pcd[y2D-1:y2D+2,x2D-1:x2D+2].reshape(-1,3)
        x3D,y3D,z3D = np.median(center_points,axis=0)

        box_rotation = np.eye(4,4)
        box_orientation = quaternion_from_matrix(box_rotation)

        detection3d = Detection3D()
        detection3d.label = detection2d.label
        detection3d.class_num = detection2d.class_num    
        detection3d.bbox3d.center.position.x = x3D
        detection3d.bbox3d.center.position.y = y3D
        detection3d.bbox3d.center.position.z = z3D
        detection3d.bbox3d.size.x = -1.0
        detection3d.bbox3d.size.y = -1.0
        detection3d.bbox3d.size.z = -1.0
        detection3d.bbox3d.center.orientation.x = box_orientation[0]
        detection3d.bbox3d.center.orientation.y = box_orientation[1]
        detection3d.bbox3d.center.orientation.z = box_orientation[2]
        detection3d.bbox3d.center.orientation.w = box_orientation[3]

        return detection3d

    def pointCloud2toArrays(self, data: PointCloud2):
        pc = ros2_numpy.numpify(data)
        if len(pc.shape) == 2:
            xyz = np.zeros((pc.shape[0], pc.shape[1], 3), dtype=np.float32)
            xyz[:, :, 0] = pc['x']
            xyz[:, :, 1] = pc['y']
            xyz[:, :, 2] = pc['z']
        elif len(pc.shape) == 1:
            xyz = np.zeros((pc.shape[0], 3), dtype=np.float32)
            xyz[:, 0] = pc['x']
            xyz[:, 1] = pc['y']
            xyz[:, 2] = pc['z']
  
        return xyz

    def arrays2toPointCloud2(self, xyz: np.ndarray, header, rgb=None):
        dtype = {'names':('x', 'y', 'z'), 'formats':('f4', 'f4', 'f4')}
        if rgb is not None:
            dtype = {'names':('x', 'y', 'z', 'rgb'), 'formats':('f4', 'f4', 'f4', 'f4')}
        
        if len(xyz.shape) == 3:
            pc = np.zeros((xyz.shape[0], xyz.shape[1]), dtype=dtype)
            pc['x'] = xyz[:, :, 0]
            pc['y'] = xyz[:, :, 1]
            pc['z'] = xyz[:, :, 2]
            if rgb is not None:
                pc['rgb'] = (rgb[:, :, 0].astype(np.uint32) << 16 | rgb[:, :, 1].astype(np.uint32) << 8 | rgb[:, :, 2].astype(np.uint32)).view(np.float32)
        elif len(xyz.shape) == 2:
            pc = np.zeros((xyz.shape[0]), dtype=dtype)
            pc['x'] = xyz[:, 0]
            pc['y'] = xyz[:, 1]
            pc['z'] = xyz[:, 2]
            if rgb is not None:
                pc['rgb'] = (rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2].astype(np.uint32)).view(np.float32)
        else:
            return None
        
        data = ros2_numpy.msgify(PointCloud2, pc, stamp=header.stamp, frame_id=header.frame_id)
        
        return data

    def detectionSeg3DArray_to_PointCloud2(self, detections3d: Detection3DArray) -> PointCloud2:
        final_pcd = PointCloud2()
        header = detections3d.header

        xyzs = []
        rgbs = []
        detection: Detection3D
        
        for detection in detections3d.detections:
            label = detection.label
            if label not in self.label_to_color:
                self.label_to_color[label] = generateRandomColor()

            xyz = self.pointCloud2toArrays(detection.mask_pcd)
            rgb = np.ones(xyz.shape, dtype=np.uint8)*self.label_to_color[label]

            xyzs.append(xyz)
            rgbs.append(rgb)
        if len(xyzs) > 0:  
            final_pcd = self.arrays2toPointCloud2(np.concatenate(xyzs, axis=0), header, rgb=np.concatenate(rgbs, axis=0))

        return final_pcd
    
    # def detectionSeg3DArray_to_MarkerArrayBbox(self, detections3d: Detection3DArray) -> MarkerArray:
    #     markers = MarkerArray()
    #     det: Detection3D
    #     for i, det in enumerate(detections3d.detections):
    #         name = det.label
    #         if name not in self.label_to_color:
    #             self.label_to_color[name] = generateRandomColor()
    #         color = self.label_to_color[name]/255.

    #         # cube marker
    #         marker = Marker()
    #         marker.header = detections3d.header
    #         marker.action = Marker.ADD
    #         marker.pose = det.bbox.center
    #         marker.color.r = color[0]
    #         marker.color.g = color[1]
    #         marker.color.b = color[2]
    #         marker.color.a = 0.4
    #         marker.ns = "bboxes"
    #         marker.id = i
    #         marker.type = Marker.CUBE
    #         marker.scale = det.bbox.size
    #         marker.lifetime = Duration(seconds=0.1).to_msg()
    #         markers.markers.append(marker)

    #         # text marker
    #         marker = Marker()
    #         marker.header = detections3d.header
    #         marker.action = Marker.ADD
    #         marker.pose = det.bbox.center
    #         marker.color.r = color[0]
    #         marker.color.g = color[1]
    #         marker.color.b = color[2]
    #         marker.color.a = 1.0
    #         marker.id = i
    #         marker.ns = "texts"
    #         marker.type = Marker.TEXT_VIEW_FACING
    #         marker.scale.x = 0.05
    #         marker.scale.y = 0.05
    #         marker.scale.z = 0.05
    #         marker.lifetime = Duration(seconds=0.1).to_msg()
    #         marker.text = '{} ({:.2f})'.format(name, det.score)
    #         markers.markers.append(marker)
        
    #     return markers

    def crop_pcd_to_marker(self, original_pcd, marker_pose, marker_scale):
        """
        Corta a nuvem de pontos original para manter apenas os pontos dentro dos limites do marcador.
        
        :param original_pcd: Nuvem de pontos original (PointCloud) do Open3D.
        :param marker_pose: Pose do marcador, contendo sua posição.
        :param marker_scale: Escala do marcador, definindo o tamanho do espaço do marcador.
        :return: Nuvem de pontos cortada.
        """
        # Converter pose e escala do marcador em limites de corte
        min_bound = marker_pose - marker_scale / 2
        max_bound = marker_pose + marker_scale / 2
        
        # Cortar a nuvem de pontos
        cropped_pcd = original_pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
        )
        
        return cropped_pcd

    def fit_model_to_marker(self, pcd, marker_bounds) -> np.ndarray:
        cropped_pcds = []
        for marker in marker_bounds.markers:
            # Extrair pontos dentro dos limites do marcador
            marker_pose = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
            marker_scale = np.array([marker.scale.x, marker.scale.y, marker.scale.z])
            
            # Cortar a nuvem de pontos com base nos limites do marcador
            cropped_pcd = self.crop_pcd_to_marker(pcd, marker_pose, marker_scale)
            cropped_pcds.append(cropped_pcd)

        combined_pcd = o3d.geometry.PointCloud()

        for temp_pcd in cropped_pcds:
            combined_pcd += temp_pcd  # Concatena o PCD atual ao PCD combinado

        return combined_pcd
    
    def createMarker(self,detection3d : Detection3D, header : Header, id) -> Marker:
        marker = Marker()
        marker.header = header
        marker.id = id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = detection3d.bbox3d.center.position.x
        marker.pose.position.y = detection3d.bbox3d.center.position.y
        marker.pose.position.z = detection3d.bbox3d.center.position.z
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        marker.scale.x = detection3d.bbox3d.size.x
        marker.scale.y = detection3d.bbox3d.size.y
        marker.scale.z = detection3d.bbox3d.size.z
        marker.lifetime = Duration(seconds=1.0).to_msg()

        return marker

    def callback(self, detections2d_msg: Detection2DArray):
        try:
            self.get_logger().info("Image 2 world: callback")
            detection3d_msg = Detection3DArray()
            detection3d_msg.header = detections2d_msg.header
            camera_info_msg = detections2d_msg.camera_info


            self.__mountLutTable(camera_info_msg)

            depth_img = self.cv_bridge.imgmsg_to_cv2(detections2d_msg.image_depth)
            depth_img = cv2.resize(depth_img, (camera_info_msg.width, camera_info_msg.height))

            pcd = np.zeros((depth_img.shape[0]*depth_img.shape[1], 3), dtype=float)

            depth_arr = depth_img.flatten()/1000
            pcd[:, 0] = self.lut_table[:, :, 0].flatten()*depth_arr # x
            pcd[:, 1] = self.lut_table[:, :, 1].flatten()*depth_arr # y
            pcd[:, 2] = depth_arr # z
            pcd = pcd.reshape((depth_img.shape[0], depth_img.shape[1], 3))
            detections3d_array: List[Detection3D] = []
            for detection2d in detections2d_msg.detections:
                for det_type in self.callbacks.keys():
                    if (detection2d.type & det_type):
                        detections3d_array.append(self.callbacks[det_type](detection2d, pcd, detections2d_msg.header))
                        break
            detection3d_msg.detections = detections3d_array
            # debug_pcd_msg = self.detectionSeg3DArray_to_PointCloud2(detection3d_msg)
            # # Publish the PointCloud2 message
            # self.pcd_publisher.publish(debug_pcd_msg)

            marker_array = MarkerArray()
            for i, detection3d in enumerate(detections3d_array):
                marker_array.markers.append(self.createMarker(detection3d, detections2d_msg.header, id=i))

            self._dbg_pub.publish(detection3d_msg)
            self.marker_publisher.publish(marker_array)
            return
        except Exception as e:
            print("Algum erro ocorreu", e)
            raise(e)
            return
            

def main(args=None):
    rclpy.init(args=args)

    image2world = Image2World()

    rclpy.spin(image2world)

    image2world.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
