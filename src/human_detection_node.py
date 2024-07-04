#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from sklearn.cluster import DBSCAN
import random

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
            distance = np.sqrt(data[0]**2 + data[1]**2 + data[2]**2)  # 거리
            if distance <= 4.0:
                points.append([data[0], data[1], data[2]])

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
                # Bounding Box Marker (테두리만 있는 박스)
                marker = Marker()
                marker.header.frame_id = "velodyne"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "human_detection"
                marker.id = id_counter
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD

                marker.scale.x = 0.1  # 선의 두께

                # 무작위 색상 설정
                marker.color.a = 1.0  # Transparency
                marker.color.r = random.random()
                marker.color.g = random.random()
                marker.color.b = random.random()

                # 박스를 구성하는 12개의 선 (8개의 점으로 구성)
                points = [
                    (x_min, y_min, z_min), (x_max, y_min, z_min),
                    (x_max, y_min, z_min), (x_max, y_max, z_min),
                    (x_max, y_max, z_min), (x_min, y_max, z_min),
                    (x_min, y_max, z_min), (x_min, y_min, z_min),
                    (x_min, y_min, z_max), (x_max, y_min, z_max),
                    (x_max, y_min, z_max), (x_max, y_max, z_max),
                    (x_max, y_max, z_max), (x_min, y_max, z_max),
                    (x_min, y_max, z_max), (x_min, y_min, z_max),
                    (x_min, y_min, z_min), (x_min, y_min, z_max),
                    (x_max, y_min, z_min), (x_max, y_min, z_max),
                    (x_max, y_max, z_min), (x_max, y_max, z_max),
                    (x_min, y_max, z_min), (x_min, y_max, z_max)
                ]

                for p in points:
                    point = [p[0], p[1], p[2], 1.0]
                    marker.points.append(Point(*point))

                markers.markers.append(marker)

                # Text Marker
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                center_z = (z_min + z_max) / 2

                text_marker = Marker()
                text_marker.header.frame_id = "velodyne"
                text_marker.header.stamp = self.get_clock().now().to_msg()
                text_marker.ns = "human_labels"
                text_marker.id = id_counter + 1000  # 고유 ID 보장
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = center_x
                text_marker.pose.position.y = center_y
                text_marker.pose.position.z = center_z + 0.5  # 박스 위에 텍스트 표시
                text_marker.pose.orientation.x = 0.0
                text_marker.pose.orientation.y = 0.0
                text_marker.pose.orientation.z = 0.0
                text_marker.pose.orientation.w = 1.0
                text_marker.scale.z = 0.3  # 텍스트 크기
                text_marker.color.a = 1.0  # Transparency
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.text = f"human{id_counter}\n({center_x:.2f}, {center_y:.2f}, {center_z:.2f})"

                markers.markers.append(text_marker)

                id_counter += 1

        self.publisher.publish(markers)
        self.get_logger().info(f'Published {len(markers.markers)//2} human markers with labels.')

    def is_human(self, x_min, x_max, y_min, y_max, z_min, z_max):
        width = x_max - x_min
        depth = y_max - y_min
        height = z_max - z_min

        # 사람의 다양한 자세와 라이다의 위치를 감안한 크기 조건 설정
        if 0.2 < width < 1.0 and 0.2 < depth < 1.0 and 0.5 < height < 2.5:
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
