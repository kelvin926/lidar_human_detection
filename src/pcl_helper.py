import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

def ros_to_pcl(ros_cloud):
    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2]])

    pcl_data = pcl.PointCloud()
    pcl_data.from_list(points_list)

    return pcl_data

def pcl_to_ros(pcl_array):
    ros_msg = PointCloud2()
    ros_msg.header.stamp = pcl_array.header.stamp
    ros_msg.header.frame_id = pcl_array.header.frame_id
    ros_msg.height = 1
    ros_msg.width = pcl_array.size
    ros_msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    ros_msg.is_bigendian = False
    ros_msg.point_step = 12
    ros_msg.row_step = ros_msg.point_step * pcl_array.size
    ros_msg.is_dense = True
    buffer = []
    for data in pcl_array:
        buffer.append(struct.pack('fff', data[0], data[1], data[2]))
    ros_msg.data = b''.join(buffer)

    return ros_msg

def do_voxel_grid_filter(pcl_data, LEAF_SIZE):
    vox = pcl_data.make_voxel_grid_filter()
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    return vox.filter()

def do_passthrough_filter(pcl_data, name_axis, axis_min, axis_max):
    passthrough = pcl_data.make_passthrough_filter()
    passthrough.set_filter_field_name(name_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    return passthrough.filter()

def get_clusters(pcl_data, tolerance, min_size, max_size):
    tree = pcl_data.make_kdtree()
    ec = pcl_data.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(tolerance)
    ec.set_MinClusterSize(min_size)
    ec.set_MaxClusterSize(max_size)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    return cluster_indices

