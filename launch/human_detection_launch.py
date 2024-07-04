from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='velodyne_driver',
            executable='velodyne_driver_node',
            name='velodyne_driver',
            parameters=[{
                'frame_id': 'velodyne',
                'model': 'VLP16'
            }],
            output='screen'
        ),
        Node(
            package='velodyne_pointcloud',
            executable='velodyne_transform_node',
            name='velodyne_transform',
            parameters=[{
                'min_range': 0.4,
                'max_range': 130.0,
                'calibration': 'config/VLP-16db.yaml',
                'model': 'VLP16'
            }],
            output='screen'
        ),
        Node(
            package='lidar_human_detection',
            executable='human_detection_node.py',
            name='human_detection_node',
            output='screen'
        ),
    ])

