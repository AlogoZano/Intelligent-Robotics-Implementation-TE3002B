import os
from launch import LaunchDescription
import launch
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('line_detection'),
        'config',
        'parameters.yaml'
    )

    microROS_agent_node  = Node(package = 'micro_ros_agent',  
                                executable='micro_ros_agent',
                                output = 'screen')
    

    odometry_launch_node  = Node(package = 'Odometry',    
                                  executable='Odemetry_Node',   
                                  output = 'screen')
    

    traffic_node  = Node(package = 'line_detection',    
                          executable='traffic_light_node',    
                          output = 'screen')
    
    line_node  = Node(package = 'line_detection',    
                        executable='line_detection_node',    
                        output = 'screen',
                        parameters = [config])
    
    
    ld = LaunchDescription ([microROS_agent_node, odometry_launch_node, traffic_node, line_node])
    return ld