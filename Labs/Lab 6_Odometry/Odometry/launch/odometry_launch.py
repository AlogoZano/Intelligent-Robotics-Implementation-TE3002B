import os
from launch import LaunchDescription
import launch
from launch_ros.actions import Node

def generate_launch_description():
    microROS_agent_node  = Node(package = 'micro_ros_agent',  
                        executable='micro_ros_agent',
                        output = 'screen')
    
    odometry_launch_node  = Node(package = 'Odometry',    
                                  executable='Odemetry_Node',   
                                  output = 'screen')
    
    teleop_node  = Node(package = 'teleop_twist_keyboard',    
                          executable='teleop_twist_keyboard',    
                          output = 'screen')
    
    ros_bag_node  = launch.actions.ExecuteProcess(
        cmd = ['ros2', 'bag', 'record', '/global_position'],
        output = 'screen'
    )
    
    ld = LaunchDescription ([microROS_agent_node, odometry_launch_node, teleop_node, ros_bag_node])
    return ld