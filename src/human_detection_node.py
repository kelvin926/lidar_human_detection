#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from sklearn.cluster import DBSCAN

class HumanDetectionNode(Node):
    def __init__(self):
        super().__init__('human_detection_node')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.pointcloud_callback,
            10)
        self.publisher = self.create_publisher(MarkerArray, '/detected_humans', 10)
        self.subscription  # prevent unused variable warning

    def pointcloud_callback(self, msg):
        self.get_logger().info(f'Received PointCloud2 data with {msg.width * msg.height} points')

        points = []
        for i, data in enumerate(pc2.read_points(msg, skip_nans=True)):
            points.append([data[0], data[1], data[2]])
            #if i < 10:
             #   self.get_logger().info(f'Point {i}: {data}')

        points = np.array(points)

        # 클러스터링
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(points)
        labels = clustering.labels_

        unique_labels = set(labels)
        self.get_logger().info(f'Found {len(unique_labels) - 1} clusters')

        markers = MarkerArray()
        id_counter = 0

        for label in unique_labels:
            if label == -1:  # 노이즈는 무시
                continue
            class_member_mask = (labels == label)
            xyz = points[class_member_mask]

            x_min = float(np.min(xyz[:, 0]))
            x_max = float(np.max(xyz[:, 0]))
            y_min = float(np.min(xyz[:, 1]))
            y_max = float(np.max(xyz[:, 1]))
            z_min = float(np.min(xyz[:, 2]))
            z_max = float(np.max(xyz[:, 2]))

            # 사람의 크기와 형태에 맞는 클러스터 필터링
            if self.is_human(x_min, x_max, y_min, y_max, z_min, z_max):
                marker = Marker()
                marker.header.frame_id = "velodyne"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "human_detection"
                marker.id = id_counter
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = (x_min + x_max) / 2
                marker.pose.position.y = (y_min + y_max) / 2
                marker.pose.position.z = (z_min + z_max) / 2
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = x_max - x_min
                marker.scale.y = y_max - y_min
                marker.scale.z = z_max - z_min
                marker.color.a = 0.5  # Transparency
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0

                markers.markers.append(marker)
                id_counter += 1

        self.publisher.publish(markers)
        self.get_logger().info(f'Published {len(markers.markers)} human markers.')

    def is_human(self, x_min, x_max, y_min, y_max, z_min, z_max):
        width = x_max - x_min
        depth = y_max - y_min
        height = z_max - z_min

        # 사람의 크기 조건: 대략적인 사람의 크기 범위를 설정
        if 0.2 < width < 1.0 and 0.2 < depth < 1.0 and 1.0 < height < 2.5:
            return True
        return False

def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

