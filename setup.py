import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'weld_seam_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'assets'), glob('assets/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Publishes a static weld scene (table, lap-joint mesh, point cloud) for RViz2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scene_publisher = weld_seam_ros2.scene_publisher_node:main',
        ],
    },
)
