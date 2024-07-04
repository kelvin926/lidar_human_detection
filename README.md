# lidar_human_detection
## Environment
ROS2 Humble
Ubuntu 22.04

## How to use
sudo apt install ros-humble-velodyne ros-humble-pcl-ros ros-humble-visualization-msgs

pip install scikit-learn

colcon build

source install/setup.bash

ros2 launch lidar_human_detection human_detection_launch.py

rviz2 -> frame_id : velodyne, ADD Point cloud2 / marker_array