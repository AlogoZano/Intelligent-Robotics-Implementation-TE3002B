#!/usr/bin/env python

# Copyright (c) 2023, Tinker Twins
# All rights reserved.

######################################################################

# ROS2 module imports
import rclpy # ROS2 client library (rcl) for Python (built on rcl C API)
from rclpy.node import Node # Node class for Python nodes
from geometry_msgs.msg import Twist # Twist (linear and angular velocities) message class
from turtlesim.msg import Pose # Turtlesim pose (x, y, theta) message class
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy # Ouality of Service (tune communication between nodes)

# Python mudule imports
from math import sqrt, atan2, pow, pi # Common mathematical functions
import random

######################################################################

# Node class
class RobotController(Node):

    #######################
    '''Class constructor'''
    #######################

    def __init__(self):
        # Information and debugging
        info = '''\nMake the robot go to the specified goal.\n'''
        print(info)
        # ROS2 infrastructure
        super().__init__('robot_controller') # Create a node with name 'robot_controller'
        qos_profile = QoSProfile( # Ouality of Service profile
        reliability=QoSReliabilityPolicy.RELIABLE, # Reliable (not best effort) communication
        history=QoSHistoryPolicy.KEEP_LAST, # Keep/store only up to last N samples
        depth=10 # Queue size/depth of 10 (only honored if the “history” policy was set to “keep last”)
        )
        self.robot_ctrl_pub = self.create_publisher(Twist, '/turtlerobot/cmd_vel', qos_profile) # Publisher which will publish Twist message to the topic '/turtle1/cmd_vel' adhering to 'qos_profile' QoS profile
        self.robot_pose_sub = self.create_subscription(Pose, '/turtlerobot/pose', self.robot_feedback_callback, qos_profile) # Subscriber which will subscribe Pose message from the topic '/turtle1/pose' and execute 'robot_feedback_callback()' callback function adhering to 'qos_profile' QoS profile
        timer_period = 0.1 # Node execution time period (seconds)
        self.timer = self.create_timer(timer_period, self.robot_controller_callback) # Define timer to execute 'robot_controller_callback()' every 'timer_period' seconds
        # Initialize variables
        self.robot_pose = Pose() # Robot pose (position & rotation)
        self.robot_flag = False # Flag to check if robot feedback is available
        self.goal_pose = Pose() # Goal pose (position & rotation)
        self.goal_flag = False # Flag to check if set goal is reached
        self.ctrl_msg = Twist() # Robot control commands (twist)
        (self.goal_pose.x, self.goal_pose.y, self.goal_pose.theta) = 2.5, 4.0, pi/2  
        self.goal_count = 0
        self.movement_type = 'linear'
        
    ########################
    '''Callback functions'''
    ########################
    
    def robot_feedback_callback(self, message):
        '''Robot feedback (pose) callback'''
        self.robot_pose = message # Capture incomming message (Pose)
        self.robot_pose.x = message.x # Extract position along x-axis
        self.robot_pose.y = message.y # Extract position along y-axis
        self.robot_pose.theta = message.theta # Extract orientation about z-axis
        self.robot_flag = True # Set robot flag to feedback available
        #print('Goal Pose  : x = {}, y = {}, theta = {}'.format(round(self.goal_pose.x, 1), round(self.goal_pose.y, 1), round(self.goal_pose.theta, 1)))
        #print('Robot Pose : x = {}, y = {}, theta = {}'.format(round(self.robot_pose.x, 1), round(self.robot_pose.y, 1), round(self.robot_pose.theta, 1)))
        #self.get_logger().info('Goal Pose  : ' + str(self.goal_pose.x) + ', ' + str(self.goal_pose.y) + ', ' + str(self.goal_pose.theta))
        #self.get_logger().info('Robot Pose : ' + str(self.robot_pose.x) + ', ' + str(self.robot_pose.y) + ', ' + str(self.robot_pose.theta))

    def robot_controller_callback(self):
        '''Robot controller (twist) callback'''
        if self.robot_flag:
            lin_vel, ang_vel = self.set_robot_control (self.movement_type) # Set and publish robot controls
            self.ctrl_msg.linear.x = lin_vel # Set robot linear velocity
            self.ctrl_msg.angular.z = ang_vel # Set robot angular velocity
            self.robot_ctrl_pub.publish(self.ctrl_msg) # Publish robot controls message
            if self.goal_flag:
                print('Robot reached specified goal pose: x = {} m, y = {} m, theta = {} rad'.format(round(self.goal_pose.x,3), round(self.goal_pose.y,3), round(self.goal_pose.theta,3)))
                quit() # Quit execution

    ######################
    '''Helper functions'''
    ######################

    def get_position_error(self):
        '''Error in position as Euclidean distance between current pose and goal pose.'''
        return sqrt(pow((self.goal_pose.x - self.robot_pose.x), 2) + pow((self.goal_pose.y - self.robot_pose.y), 2))
    
    def get_rotation_error(self):
        '''Error in rotation as relative angle between current pose and goal pose.'''
        desired_angle = atan2(self.goal_pose.y - self.robot_pose.y, self.goal_pose.x - self.robot_pose.x)
        dtheta = desired_angle - self.robot_pose.theta + 2*pi
        if abs(desired_angle - self.robot_pose.theta) < abs(desired_angle - self.robot_pose.theta + 2*pi):
            dtheta = desired_angle - self.robot_pose.theta
        return dtheta

    def get_linear_velocity(self, gain=1.0):
        '''Compute robot linear velocity using P-controller'''
        return gain * self.get_position_error()

    def get_angular_velocity(self, gain=1.0):
        '''Compute robot angular velocity using P-controller'''
        return gain * self.get_rotation_error()

    def set_robot_control(self, movement_type):
        rot_tol = 0.005
        pos_tol = 0.05
        if movement_type == 'angular':
            if abs(self.get_rotation_error()) > rot_tol: # Orient at goal   
                lin_vel = 0.0 # Set robot linear velocity
                ang_vel = self.get_angular_velocity(gain=2.0) # Set robot angular velocity
                return lin_vel, ang_vel 
            else:
                self.set_robot_sequence()
        if movement_type == 'linear':
            if self.get_position_error() > pos_tol: # Go to goal  
                lin_vel = self.get_linear_velocity(gain=1.0) # Set robot linear velocity
                ang_vel = self.get_angular_velocity(gain=4.0) # Set robot angular velocity
                return lin_vel, ang_vel
            else:
                self.set_robot_sequence()
            
            # Stop robot after the goal is reached
        lin_vel = 0.0 # Set robot linear velocity
        ang_vel = 0.0 # Set robot angular velocity
        return lin_vel, ang_vel
   
    def set_robot_sequence(self):
        # Set goal pose
        # CUADRADO
        # self.seq_data = [['linear', 10.0, 1.0,    0.0],
        #                  ['angular', 10.0, 10.0,   pi/2],
        #                  ['linear', 10.0, 10.0,   pi/2],
        #                  ['angular', 1.0,  10.0,   pi],
        #                  ['linear', 1.0,  10.0,   pi],
        #                  ['angular', 1.0,  1.0,    3*pi/2],
        #                  ['linear', 1.0,  1.0,    3*pi/2],
        #                  ['angular', 10.0, 1.0,    0.0],]
        
        # RECTÁNGULO

        self.seq_data = [['linear', 2.5, 8.0,    pi/2],
                         ['angular', 8.5, 8.0,   pi/2],
                         ['linear', 8.5, 8.0,   pi/2],
                         ['angular', 8.5,  4.0,   3*pi/2],
                         ['linear', 8.5,  4.0,   3*pi/2],
                         ['angular', 2.5,  4.0,    pi],
                         ['linear', 2.5,  4.0,    pi],
                         ]

        # CURVA
        # self.seq_data = [['linear', 5.5, 10.0, pi/2]]

        
        if self.goal_count < len(self.seq_data):
            (self.movement_type, self.goal_pose.x, self.goal_pose.y, self.goal_pose.theta) = self.seq_data[self.goal_count]
            print(self.movement_type, self.goal_pose.x, self.goal_pose.y, self.goal_pose.theta)
            self.goal_count += 1
            print("goal count: ", self.goal_count)
            return self.goal_pose.x, self.goal_pose.y, self.goal_pose.theta
                
        self.goal_flag = True # Set goal flag to reached
        
######################################################################

def main(args=None):
    rclpy.init(args=args) # Start ROS2 communications
    node = RobotController() # Create node
    rclpy.spin(node) # Execute node
    node.destroy_node() # Destroy node explicitly (optional - otherwise it will be done automatically when garbage collector destroys the node object)
    rclpy.shutdown() # Shutdown ROS2 communications

######################################################################

if __name__ == "__main__":
    main()