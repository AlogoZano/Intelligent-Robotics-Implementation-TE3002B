import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, Int32, Int8
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import Image
import rclpy.qos
import cv2 
import skfuzzy as sk
from cv_bridge import CvBridge


class fuzzy_controller_node(Node):

    def __init__(self):
        super().__init__('fuzzy_controller_node')
    
        self.publisher_cmd_vel = self.create_publisher(Twist, 'cmd_vel', qos_profile=rclpy.qos.qos_profile_sensor_data)
       
        self.sub_line_error = self.create_subscription(Int32, 
                                            'line_error', 
                                            self.fuzzy_control_callback,
                                            10)
        self.sub_cross_walk_flag = self.create_subscription(Bool, 
                                            'cross_walk_flag', 
                                            self.cross_walk_flag_callback,
                                            10)
        self.sub_red_light = self.create_subscription(Float32, 
                                    'traffic_light/red/density', 
                                    self.red_light_callback,
                                    10)
        self.sub_green_light = self.create_subscription(Float32, 
                                            'traffic_light/green/density', 
                                            self.green_light_callback,
                                            10)
        
        self.sub_sign = self.create_subscription(Int8, 
                    'class_name', 
                    self.signals_callback,
                    10)
        
        
        self.msg_control = Twist()
        
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        ###### variables ######
        self.bridge_object = CvBridge()
        self.v_ang = 0.0
        self.signal_received = 0
        self.speed_const = 1.0
        
        self.cross_walk_flag = False
        self.line_error = 0
        
        ## fuzzy logic limits##
        
        self.linea = np.arange(-356, 357, 1)
        self.vel_ang = np.arange(-0.562, 0.563, 0.01)
        self.v_linear = 0.0

        # Membership functions for the input variable 'linea'
        self.linea_der = sk.trapmf(self.linea, [-356,-356,-240,-210])
        self.linea_mid_der= sk.trapmf(self.linea, [-220,-180,-80,-40])
        self.linea_centro = sk.trapmf(self.linea, [-60,-20,20,60])
        self.linea_mid_izq = sk.trapmf(self.linea, [40,80,180,220])
        self.linea_izq = sk.trapmf(self.linea, [210,240,357,357])
  
        self.v_ang_izq = sk.trimf(self.vel_ang, [-0.562, -0.5, -0.312])
        self.v_ang_med_izq = sk.trimf(self.vel_ang, [-0.435, -0.25, -0.062])
        self.v_ang_centro = sk.trimf(self.vel_ang, [-0.187, 0.0, 0.187])
        self.v_ang_med_der = sk.trimf(self.vel_ang, [0.0625, 0.25, 0.435])
        self.v_ang_der = sk.trimf(self.vel_ang, [0.312, 0.5, 0.562])
        
        self.v_angular = 0.0
        
        # densities ##
        self.green_density = 0.0
        self.red_density = 0.0
        
        # flags #
        self.crossWalk_flag = False
        self.traffic_flag = False
        
        ###### logger ######
        self.get_logger().info('fuzzy CONTROL')

    def green_light_callback(self,msg):
        self.green_density = msg.data 
        if self.green_density > 0.15:
            self.traffic_flag = True
            self.hor_flag = False

    def red_light_callback(self,msg):
        self.red_density = msg.data
        if self.red_density > 0.15:
            self.traffic_flag = False
    
    def cross_walk_flag_callback(self, msg):
        self.cross_walk_flag = msg.data 

    def signals_callback(self, msg):
        self.signal_received = msg.data
    
    def fuzzy_control_callback(self, msg):

        self.line_error = msg.data
        self.line_error += 357
        print("error: ", self.line_error)

        R1 = self.linea_izq[self.line_error]
        R2 = self.linea_mid_izq[self.line_error]
        R3 = self.linea_centro[self.line_error]
        R4 = self.linea_mid_der[self.line_error]
        R5 = self.linea_der[self.line_error]
        
        # Inference (using min operator for 'and')
        out_izq = np.fmin(R1, self.v_ang_der)
        out_mid_izq = np.fmin(R2, self.v_ang_med_der)
        out_mid = np.fmin(R3, self.v_ang_centro)
        out_mid_der = np.fmin(R4, self.v_ang_med_izq)
        out_der = np.fmin(R5, self.v_ang_izq)

        # Aggregation (using max operator for 'or')
        aggregated = np.fmax(out_izq,
                            np.fmax(out_mid_izq,
                                    np.fmax(out_mid,
                                            np.fmax(out_mid_der, out_der))))
        
        if self.traffic_flag == True and self.hor_flag == False:
            self.v_linear = 0.02

            if self.signal_received == 4:
                self.speed_const = 0.5

            elif self.signal_received == 5:
                self.speed_const = 1.5

            elif self.signal_received == 6:
                self.speed_const = 0.0


            self.v_linear = self.v_linear*self.speed_const
            self.v_angular = (sk.defuzz(self.vel_ang, aggregated, 'centroid'))*(-0.1)*self.speed_const
            
            print("v_ang:",self.v_angular)

            
        else:
            self.v_angular = 0.0
            self.v_linear = 0.0


    def timer_callback(self):
        
        self.msg_control.angular.z = self.v_angular
        self.msg_control.linear.x = self.v_linear
        self.publisher_cmd_vel.publish(self.msg_control)


def main(args = None):
    rclpy.init(args=args)
    fuzzy_vAng = fuzzy_controller_node()
    rclpy.spin(fuzzy_vAng) #While(1)
    fuzzy_vAng.destroy_node() #Destruir nodo
    rclpy.shutdown()    

if __name__ == '__main__':
    main()