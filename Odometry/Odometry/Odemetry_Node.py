import rclpy
import math as ma
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose2D
import rclpy.qos

class Odometry_node(Node):

    def __init__(self):
        super().__init__('Odometry_node')

        self.publisher_displacement = self.create_publisher(Float32, 'displacement', qos_profile=rclpy.qos.qos_profile_sensor_data) #displacement
        self.publisher_Speed = self.create_publisher(Float32, 'linear_speed', qos_profile=rclpy.qos.qos_profile_sensor_data) #linear_speed
        self.publisher_GlobalPosition = self.create_publisher(Pose2D, 'global_position', qos_profile=rclpy.qos.qos_profile_sensor_data) #coordinates
        #self.publisher_theta = self.create_publisher(Float32, 'theta', qos_profile=rclpy.qos.qos_profile_sensor_data) #coordinates
        self.publisher_AngularSpeed = self.create_publisher(Float32, 'angular_speed', qos_profile=rclpy.qos.qos_profile_sensor_data) #angular speed w

        self.msg_displacement = Float32()
        self.msg_speed = Float32()
        self.msg_globalP = Pose2D()
        self.msg_angularS = Float32()
        self.msg_Orientation = Float32()

        self.msg_theta = Float32()

        self.timer_period = 0.1
        self.time = 0

        self.angSR = 0.0
        self.angSL = 0.0
        self.linear_speed = 0.0
        self.posx = 0.0
        self.posy = 0.0
        self.theta = 0.0
        self.displacement = 0.0
        self.ang_speed = 0.0

        self.radius = 0.05
        self.length  = 0.18

        self.previous_angS  = 0.0
        self.previous_linearS = 0.0

        

        self.angSpeedRight = self.create_subscription(Float32, 
                                            'VelocityEncR', 
                                            self.listener_angSR_callback,
                                            qos_profile=rclpy.qos.qos_profile_sensor_data)
        self.angSpeedLeft = self.create_subscription(Float32, 
                                            'VelocityEncL', 
                                            self.listener_angSL_callback,
                                            qos_profile=rclpy.qos.qos_profile_sensor_data)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info('Odometry_Dash node successfully initialized!')

    def timer_callback(self):
        
        #Linear and angular speed obtention
        self.linear_speed = self.radius * (self.angSL + self.angSR)/2
        self.ang_speed = self.radius * (self.angSR - self.angSL) / self.length

        #Integrating velocity we obtain global displacement and orientation (theta)
        self.theta += (self.ang_speed) * self.timer_period
        self.displacement += (self.linear_speed) * self.timer_period 

        #Obtaining Vx and Vy
        vx = self.linear_speed*ma.cos(self.theta)
        vy = self.linear_speed*ma.sin(self.theta)

        #Integrating Vx and Vy we obtain the global position 
        self.posx += (vx) * self.timer_period
        self.posy += (vy) * self.timer_period 

        #Assign obtained values to message data
        self.msg_speed.data = self.linear_speed
        self.msg_angularS.data = self.ang_speed
        self.msg_displacement.data = self.displacement
        self.msg_globalP.x = self.posx
        self.msg_globalP.y = self.posy
        self.msg_globalP.theta = self.theta

        #Publish messages
        self.publisher_Speed.publish(self.msg_speed)
        self.publisher_AngularSpeed.publish(self.msg_angularS)
        self.publisher_displacement.publish(self.msg_displacement)
        self.publisher_GlobalPosition.publish(self.msg_globalP)
        
    def listener_angSR_callback (self, msg):
        self.angSR = msg.data

    def listener_angSL_callback (self, msg):
        self.angSL = msg.data

        

def main(args = None):
    rclpy.init(args=args)
    odometry = Odometry_node()
    rclpy.spin(odometry) #While(1)
    odometry.destroy_node() #Destruir nodo
    rclpy.shutdown()    

if __name__ == '__main__':
    main()