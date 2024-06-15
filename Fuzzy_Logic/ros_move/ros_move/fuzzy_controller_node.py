import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32
import rclpy.qos
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from math import sqrt, atan2
import csv

class Fuzzy_Controller(Node):

    def __init__(self):
        super().__init__('fuzzy_controller_node')

        qos_profile = QoSProfile( 
            reliability=QoSReliabilityPolicy.RELIABLE, 
            history=QoSHistoryPolicy.KEEP_LAST, 
            depth=10 
        )
        self.robot_ctrl_pub = self.create_publisher(Twist, '/turtlerobot/cmd_vel', qos_profile)
        self.robot_pose_sub = self.create_subscription(Pose, '/turtlerobot/pose', self.robot_feedback_callback, qos_profile)
        self.bot_pose_sub = self.create_subscription(Pose, '/turtlebot/pose', self.bot_feedback_callback, qos_profile)

        ##timer
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.robot_pose = Pose()
        self.bot_pose = Pose()
        self.ctrl_msg = Twist()

        self.distances = []
        self.orientations = []
        self.lookup_table = {}
        with open('/home/alogo/ros2_ws/src/ros_move/ros_move/lookup_table.csv', mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                dist = float(row["distance"])
                orient = float(row["orientation"])
                vel_lin = float(row["vel_lin"])
                vel_ang = float(row["vel_ang"])
                if dist not in self.distances:
                    self.distances.append(dist)
                if orient not in self.orientations:
                    self.orientations.append(orient)
                self.lookup_table[(dist, orient)] = (vel_lin, vel_ang)

        self.distances = sorted(self.distances)
        self.orientations = sorted(self.orientations)

    def robot_feedback_callback(self, msg):
        self.robot_pose = msg
        self.robot_pose.x = msg.x
        self.robot_pose.y = msg.y
        self.robot_pose.theta = msg.theta

    def bot_feedback_callback(self, msg):
        self.bot_pose = msg
        self.bot_pose.x = msg.x
        self.bot_pose.y = msg.y
        self.bot_pose.theta = msg.theta

    def get_distance_error(self):
        return sqrt(pow((self.bot_pose.x - self.robot_pose.x), 2) + pow((self.bot_pose.y - self.robot_pose.y), 2))

    def get_orientation_error(self):
        return self.robot_pose.theta - (atan2(self.bot_pose.y - self.robot_pose.y, self.bot_pose.x - self.robot_pose.x))

    def find_closest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def controller_routine(self):
        distance = round(self.get_distance_error(),3)
        orientation = round(self.get_orientation_error(),3)

        msg = "Distancia: {}. Orientación: {}".format(distance, orientation)
        self.get_logger().info(msg)

        closest_dist = self.find_closest(self.distances, distance)
        closest_orient = self.find_closest(self.orientations, orientation)

        return self.lookup_table.get((closest_dist, closest_orient), (0.0, 0.0))

    def timer_callback(self):
        vel_lin, vel_ang = self.controller_routine()
        msg = "Linear: {}. Angular: {}".format(vel_lin, vel_ang)
        self.ctrl_msg.linear.x = vel_lin*0.2
        self.ctrl_msg.angular.z = -vel_ang*0.5

        self.robot_ctrl_pub.publish(self.ctrl_msg)

        self.get_logger().info(msg)

def main(args=None):
    rclpy.init(args=args)
    fuzzy = Fuzzy_Controller()
    rclpy.spin(fuzzy)
    fuzzy.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
