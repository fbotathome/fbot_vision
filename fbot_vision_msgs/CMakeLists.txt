cmake_minimum_required(VERSION 3.8)
project(fbot_vision_msgs)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED) 
find_package(vision_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	add_compile_options(-Wall -Wextra -Wpedantic)
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KeyPoint2D.msg"
  "msg/KeyPoint3D.msg"
  "msg/Detection2D.msg"
  "msg/Detection2DArray.msg"
  "msg/Detection3D.msg"
  "msg/Detection3DArray.msg"
  "msg/FaceDetection3D.msg"
  "msg/FaceDetection3DArray.msg"
  "msg/VLMQuestion.msg"
  "msg/VLMAnswer.msg"
  "msg/PersonDetection.msg"
	"msg/PersonDetectionList.msg"
  "srv/PeopleIntroducing.srv"
  "srv/LookAtDescription3D.srv"
  "srv/VLMQuestionAnswering.srv"
  "srv/VLMAnswerHistory.srv"
  "srv/PeopleForgetting.srv"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs vision_msgs
)

# Export runtime dependencies
ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(std_msgs geometry_msgs sensor_msgs vision_msgs nav_msgs)

ament_package()