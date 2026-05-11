from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'stretch4_hybrid_marker_demos'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hello Robot Inc.',
    maintainer_email='support@hello-robot.com',
    description='Demonstrations of using hybrid markers with the Stretch 4 mobile manipulator.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros_pursue_target = stretch4_hybrid_marker_demos.ros_pursue_target:main',
            'ros_track_object = stretch4_hybrid_marker_demos.ros_track_object:main'
        ],
    },
)
