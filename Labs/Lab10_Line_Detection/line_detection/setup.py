from setuptools import find_packages, setup

package_name = 'line_detection'

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
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_detection_node = line_detection.line_detection_node:main',
            'traffic_light_node = line_detection.traffic_light_node:main'
        ],
    },
)
