# human_detection_package
## Environment
ROS2 Humble
Ubuntu 22.04

## How to use
sudo apt install ros-humble-velodyne ros-humble-pcl-ros ros-humble-visualization-msgs

pip install scikit-learn

colcon build

ros2 launch human_detection_package velodyne_launch.py

## Important
velodyne_launch.py - 23 lines
'calibration': '/home/hyunseo/human_detection_package/config/VLP-16db.yaml', -> your absolute path
