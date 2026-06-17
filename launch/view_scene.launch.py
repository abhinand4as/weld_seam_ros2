#!/usr/bin/env python3
"""Launch the scene publisher and RViz2.

Publishes the static weld scene (table mesh, lap-joint mesh, point cloud)
and opens RViz2 with the preconfigured display layout.  Run the seam
detector separately via detector.launch.py.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('weld_seam_ros2')
    rviz_config = os.path.join(pkg_share, 'rviz', 'scene.rviz')

    # RViz's Fixed Frame must exist in TF; message headers alone don't
    # register a frame, so broadcast one static identity transform.
    world_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_tf',
        arguments=['--frame-id', 'world', '--child-frame-id', 'table'],
    )

    scene_publisher = Node(
        package='weld_seam_ros2',
        executable='scene_publisher',
        name='scene_publisher',
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    return LaunchDescription([world_tf, scene_publisher, rviz])
