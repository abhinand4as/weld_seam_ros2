#!/usr/bin/env python3
"""Launch the weld-seam detector node.

Launch argument
---------------
detector : 'default' (default) | 'lap_joint'

    default    — seam_detector
                 Full asymmetry-feature + B-spline pipeline. Suitable for
                 V-groove and T-joint geometries.

    lap_joint  — lap_joint_seam_detector
                 Height-split / XY-proximity pipeline. Designed specifically
                 for lap joints where the two-plate Z bimodality is exploited
                 directly, avoiding normal estimation on flat surfaces.

Examples
--------
ros2 launch weld_seam_ros2 detector.launch.py
ros2 launch weld_seam_ros2 detector.launch.py detector:=lap_joint
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_EXECUTABLES = {
    'default':   'seam_detector',
    'lap_joint': 'lap_joint_seam_detector',
}


def _launch_setup(context, *args, **kwargs):
    detector = LaunchConfiguration('detector').perform(context)
    executable = _EXECUTABLES.get(detector, 'seam_detector')
    return [
        Node(
            package='weld_seam_ros2',
            executable=executable,
            name='seam_detector',
            output='screen',
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'detector',
            default_value='default',
            description=(
                "Detector variant: 'default' (asymmetry+B-spline) "
                "or 'lap_joint' (height-split)"
            ),
        ),
        OpaqueFunction(function=_launch_setup),
    ])
