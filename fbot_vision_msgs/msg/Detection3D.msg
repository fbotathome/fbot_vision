std_msgs/Header header
std_msgs/Header poses_header

int64 id #local id on image (can come from detection or instance segmentation) (optional)
int64 global_id #global id to identify a description (optional)
string label #class detected by classification, detection, semantic segmentation or instance segmentation 
int64 class_num 
float64 score #score of the description given by the algorithm

vision_msgs/BoundingBox3D bbox3d #3D bounding box of the description
vision_msgs/BoundingBox2D bbox2d #2D bounding box of the description
std_msgs/ColorRGBA mean_color #mean color of the Description3D

fbot_vision_msgs/KeyPoint3D[] pose #keypoints array

string uuid #unique identifier for the person of the detected face 
