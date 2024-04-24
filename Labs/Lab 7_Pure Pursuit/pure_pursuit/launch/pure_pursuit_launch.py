import os
from launch import LaunchDescription
import launch
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('pure_pursuit'),
        'config',
        'parameters.yaml'
    )

    microROS_agent_node  = Node(package = 'micro_ros_agent',  
                                executable='micro_ros_agent',
                                output = 'screen')
    
    odometry_launch_node  = Node(package = 'Odometry',    
                                  executable='Odemetry_Node',   
                                  output = 'screen')
    
    pp_node  = Node(package = 'pure_pursuit',    
                          executable='pure_pursuit_node',    
                          output = 'screen',
                          parameters = [config])
    
    
    ld = LaunchDescription ([microROS_agent_node, odometry_launch_node, pp_node])
    return ld