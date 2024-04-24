import rclpy
import math as ma
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose2D, Twist
import rclpy.qos

class pure_pursuit_node(Node):

    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('ld', rclpy.Parameter.Type.DOUBLE),
                ('vlin', rclpy.Parameter.Type.DOUBLE),
                ('radius', rclpy.Parameter.Type.DOUBLE),
                ('straight.xref', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('straight.yref', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('lane_change.xref', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('lane_change.yref', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('square.xref', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('square.yref', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('triangle.xref', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('triangle.yref', rclpy.Parameter.Type.DOUBLE_ARRAY),
            ]
        )

        self.publisher_cmd_vel = self.create_publisher(Twist, 'cmd_vel', qos_profile=rclpy.qos.qos_profile_sensor_data) 
        self.publisher_k = self.create_publisher(Float32, 'k', qos_profile=rclpy.qos.qos_profile_sensor_data) 

        self.msg_control = Twist()

        self.xref = self.get_parameter('lane_change.xref').get_parameter_value().double_array_value
        self.yref = self.get_parameter('lane_change.yref').get_parameter_value().double_array_value
        
        self.ld = self.get_parameter('ld').get_parameter_value().double_value
        self.L = 0.18

        self.wp_distance = 0.0

        self.alpha = 0.0
        self.delta = 0.0
        self.msg_k = Float32()
        self.k = 0.0

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.vlin = self.get_parameter('vlin').get_parameter_value().double_value
        self.vang = 0.0

        self.i = 0

        self.timer_period = 0.01

        self.radius = self.get_parameter('radius').get_parameter_value().double_value
        

        self.pose_2d = self.create_subscription(Pose2D, 
                                            'global_position', 
                                            self.listener_pose_callback,
                                            qos_profile=rclpy.qos.qos_profile_sensor_data)
       
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info('PURA DOBLE P VIEJONES!')

    def calculate_distance(self, delta_x, delta_y):
        return np.sqrt((delta_x)**2 + (delta_y)**2)

    def next_waypoint(self, x, y, i):
        for i in range(i, len(self.xref)):
            deltax = self.xref[i] - x
            deltay = self.yref[i] - y
            distance = self.calculate_distance(deltax, deltay)

            if distance >= self.ld:
                return i
            
            elif i == len(self.xref) - 1:
                return i
            
        return i

    
    def listener_pose_callback (self, msg):

        self.x = msg.x
        self.y = msg.y
        self.theta = msg.theta

        self.i = self.next_waypoint(self.x, self.y, self.i)


        delta_travel_x = self.x - self.xref[self.i]
        delta_travel_y = self.y - self.yref[self.i]

        travel_distance = self.calculate_distance(delta_travel_x, delta_travel_y)

        if travel_distance > self.radius:
            self.alpha = np.arctan2(delta_travel_y, delta_travel_x) - self.theta
            self.k = -(2*np.sin(self.alpha))/self.ld
            #self.delta = np.arctan(self.k * self.L)
            self.vang = self.k * self.get_parameter('vlin').get_parameter_value().double_value

        else:
            if self.i == len(self.xref)-1:
                self.vang = 0.0
                self.vlin = 0.0

            else:
                self.i = self.i
            
    
    def timer_callback(self):

        self.msg_control.angular.z = self.vang
        self.msg_control.linear.x = self.vlin
        self.msg_k.data = self.k
        #Publish messages
        self.publisher_cmd_vel.publish(self.msg_control)
        self.publisher_k.publish(self.msg_k)
        


        
def main(args = None):
    rclpy.init(args=args)
    pure_pursuit = pure_pursuit_node()
    rclpy.spin(pure_pursuit) #While(1)
    pure_pursuit.destroy_node() #Destruir nodo
    rclpy.shutdown()    

if __name__ == '__main__':
    main()