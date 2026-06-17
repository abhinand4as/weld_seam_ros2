#!/usr/bin/env python3
"""
scene_publisher_node.py — publish a static weld inspection scene for RViz2.

Publishes three things in the same fixed frame so they line up visually:
  - a table (Marker CUBE, generated — no mesh file needed)
  - the lap-joint test part (Marker MESH_RESOURCE, assets/lap_joint_test.stl)
  - the captured point cloud of that part (PointCloud2, assets/*.pcd)

The part's local origin sits with its bottom face at Z=0 in both the STL
and the default PCD, so placing the table top at Z=0 makes the cloud and
mesh overlap correctly on top of the table.

Topics
------
/scene/markers   visualization_msgs/MarkerArray   table + part mesh
/scene/points    sensor_msgs/PointCloud2          part point cloud
"""
import os

import numpy as np
import open3d as o3d
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]


class ScenePublisher(Node):
    def __init__(self):
        super().__init__('scene_publisher')

        share_dir = get_package_share_directory('weld_seam_ros2')
        default_mesh = os.path.join(share_dir, 'assets', 'lap_joint_test.stl')
        default_pcd = os.path.join(share_dir, 'assets', 'lap_joint_test.pcd')

        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('mesh_path', default_mesh)
        self.declare_parameter('pcd_path', default_pcd)
        self.declare_parameter('table_width', 0.6)    # X (m)
        self.declare_parameter('table_depth', 0.6)    # Y (m)
        self.declare_parameter('table_height', 0.05)  # Z (m), top surface at Z=0
        self.declare_parameter('publish_rate', 2.0)   # Hz

        self._frame_id = self.get_parameter('frame_id').value
        self._table_width = self.get_parameter('table_width').value
        self._table_depth = self.get_parameter('table_depth').value
        self._table_height = self.get_parameter('table_height').value

        mesh_path = self.get_parameter('mesh_path').value
        self._mesh_resource = 'file://' + mesh_path

        pcd_path = self.get_parameter('pcd_path').value
        pcd = o3d.io.read_point_cloud(pcd_path)
        self._points = np.asarray(pcd.points, dtype=np.float32)
        if len(self._points) == 0:
            self.get_logger().error(f'Empty PCD: {pcd_path}')
            raise RuntimeError('Empty PCD')

        self.get_logger().info(
            f'Loaded {len(self._points)} points from {pcd_path}\n'
            f'  mesh: {mesh_path}\n'
            f'  frame_id: {self._frame_id}'
        )

        self._marker_pub = self.create_publisher(MarkerArray, '/scene/markers', 10)
        self._cloud_pub = self.create_publisher(PointCloud2, '/scene/points', 10)

        rate = self.get_parameter('publish_rate').value
        self._timer = self.create_timer(1.0 / rate, self._publish)

    def _header(self) -> Header:
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = self._frame_id
        return h

    def _table_marker(self) -> Marker:
        m = Marker()
        m.header = self._header()
        m.ns = 'scene'
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.z = -self._table_height / 2.0
        m.pose.orientation.w = 1.0
        m.scale.x = self._table_width
        m.scale.y = self._table_depth
        m.scale.z = self._table_height
        m.color = ColorRGBA(r=0.55, g=0.4, b=0.25, a=1.0)
        return m

    def _part_mesh_marker(self) -> Marker:
        m = Marker()
        m.header = self._header()
        m.ns = 'scene'
        m.id = 1
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD
        m.mesh_resource = self._mesh_resource
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color = ColorRGBA(r=0.7, g=0.7, b=0.75, a=1.0)
        return m

    def _publish(self):
        markers = MarkerArray()
        markers.markers.append(self._table_marker())
        markers.markers.append(self._part_mesh_marker())
        self._marker_pub.publish(markers)

        cloud = pc2.create_cloud(self._header(), FIELDS_XYZ, self._points.tolist())
        self._cloud_pub.publish(cloud)


def main():
    rclpy.init()
    node = ScenePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
