import os
from launch import LaunchDescription
import launch
from launch_ros.actions import Node

def generate_launch_description():

    visualize_rqt  = Node(package = 'rqt_image_view',  
                    executable='rqt_image_view',
                    output = 'screen')
    
    video_source  = Node(package = 'ros_deep_learning',  
                    executable='video_viewer.ros2.launch',
                    output = 'screen')
    
    
    color_identification  = Node(package = 'color_identification',    
                                  executable='color_identification_node',   
                                  output = 'screen')
        
    ld = LaunchDescription ([video_source, color_identification, visualize_rqt])
    return ld