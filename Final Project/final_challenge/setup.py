from setuptools import find_packages, setup

package_name = 'final_challenge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'line_follower_node = final_challenge.line_follower_node:main',
            'traffic_light_node = final_challenge.traffic_light_node:main',
            'path_detection_node = final_challenge.path_detection_node:main',
            'fuzzy_controller_node = final_challenge.fuzzy_controller_node:main',
            'process_img_node = final_challenge.process_img_node:main',
        ],
    },
)
