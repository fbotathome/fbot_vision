cmake_minimum_required(VERSION 3.22)
project(fbot_recognition)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	add_compile_options(-Wall -Werror -Wno-attributes)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(fbot_vision_msgs REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Development.Embed NumPy)
find_package(pybind11 REQUIRED)
find_package(OnnxRuntime REQUIRED)

add_library(lasermodelhost SHARED
	src/lasermodelhost.cpp
	src/lasermodelhost_follower.cpp
)

target_compile_features(lasermodelhost PUBLIC c_std_99 cxx_std_17) # Require C99 and C++17

target_link_libraries(lasermodelhost PRIVATE
	OnnxRuntime::Lib
	pybind11::embed
	Python3::Python
)

ament_target_dependencies(lasermodelhost PUBLIC
	rclcpp
	rclcpp_components
	std_msgs
	sensor_msgs
	visualization_msgs
	geometry_msgs
	fbot_vision_msgs
)

rclcpp_components_register_node(lasermodelhost
	PLUGIN "laser_people_detector::LaserModelHost"
	EXECUTABLE lasermodelnode
)

rclcpp_components_register_node(lasermodelhost
	PLUGIN "laser_people_detector::LaserModelHostFollower"
	EXECUTABLE lasermodelnode_follower
)

install(
	TARGETS lasermodelhost
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
)

install(DIRECTORY
	launch
	config
	resource
	DESTINATION share/${PROJECT_NAME}
)

ament_package()
