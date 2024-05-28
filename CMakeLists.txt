cmake_minimum_required(VERSION 3.5)
project(my_human_detection_package)

# Add the following lines to set the policy CMP0074
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common io)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

install(PROGRAMS
  src/human_detection_node.py
  src/pcl_helper.py
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()

