import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
#import sys
import math
#import time

class GoToGoal(Node):

    x = 0.0
    y = 0.0
    theta = 0.0
    xgoal = 0.0
    ygoal = 0.0
    
    def __init__(self):
        super().__init__("go_to_goal")
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)
        self.timer = self.create_timer(0.1, self.go_to_goal)
        #self.pose = Pose()
        self.set_goal()
    
    def set_goal(self):
        global xgoal
        global ygoal
        xgoal = 10.0 #float(input("Enter a desired x position: "))
        ygoal = 10.0 #float(input("Enter a desired y position: "))

    def pose_callback(self, pose_msg = Pose()): #(self, data)
        global x
        global y
        global theta
        x = pose_msg.x
        y = pose_msg.y
        theta = pose_msg.theta
    #    print("pose:", round(x,3), round(y,3), round(theta,3))
            
    def go_to_goal(self):
        global x
        global y
        global theta
        global xgoal
        global ygoal
        dist_x = xgoal - x
        dist_y = ygoal - y
        print(dist_x)
        euclidean_distance = math.sqrt(dist_x**2 + dist_y**2)

        d_tol = 0.2
        a_tol = 0.05

        speed = Twist()
        if (euclidean_distance > d_tol):
            kv = 1.5				
            speed.linear.x = kv * euclidean_distance

            ka = 2.0
            dtheta = 0.0
            desired_angle_goal = math.atan2(dist_y, dist_x)
            
            dtheta = 0.5 #modificar
            
            if dtheta > a_tol:
                speed.angular.z = ka * (dtheta)
            else:
                speed.angular.z = 0.0
            
        else:
            speed.linear.x = 0.0
            self.get_logger().info("Target reached")
            self.timer.cancel()

        self.cmd_vel_pub.publish(speed)

def main(args=None):
    rclpy.init(args=args)
    go_to_goal = GoToGoal()
    rclpy.spin(go_to_goal)
    rclpy.shutdown()

if __name__ == '__main__':
    main()