uint8 DETECTION=0
uint8 INSTANCE_SEGMENTATION=1
uint8 SEMANTIC_SEGMENTATION=2
uint8 CLASSIFICATION=3
uint8 POSE=4
uint8 TRACKING=5

std_msgs/Header header

uint8 type

int64 id #local id on image (can come from detection or instance segmentation) (optional)
int64 global_id #global id to identify a description (optional)
string label #class detected by classification, detection, semantic segmentation or instance segmentation 
int64 class_num
geometry_msgs/Vector3 max_size #max 3D size of a object, can be used to filter 3d bounding box 
float64 score #score of the description given by the algorithm

sensor_msgs/Image mask #segmentation mask (optional)
vision_msgs/BoundingBox2D bbox #(optional)
fbot_vision_msgs/KeyPoint2D[] pose