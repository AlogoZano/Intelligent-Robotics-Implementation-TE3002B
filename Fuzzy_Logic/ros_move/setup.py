from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ros_move'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch','*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alogo',
    maintainer_email='alogo@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "rosmove_node = ros_move.rosmove_node:main",
            "square_node = ros_move.square_node:main",
            "quasisquare_node = ros_move.quasisquare_node:main",
            "gotogoal_node = ros_move.gotogoal_node:main",
            "speed_change_node = ros_move.speed_change_node:main",
            "bot_node = ros_move.bot_node:main",
            "lane_change_node = ros_move.lane_change_node:main",
            "fuzzy_controller_node = ros_move.fuzzy_controller_node:main",
        ],
    },
)
