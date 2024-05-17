import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import Image
import rclpy.qos
import cv2 
from cv_bridge import CvBridge


class line_detection_node(Node):

    def __init__(self):
        super().__init__('line_detection_node')

        
        self.publisher_img_prop_frame = self.create_publisher(Image, 'img_properties/frame', 10)
        self.publisher_cmd_vel = self.create_publisher(Twist, 'cmd_vel', qos_profile=rclpy.qos.qos_profile_sensor_data)
       
        self.sub_video = self.create_subscription(Image, 
                                            'video_source/raw', 
                                            self.line_detection_callback,
                                            10)
        self.sub_red_light = self.create_subscription(Float32, 
                                            'traffic_light/red/density', 
                                            self.red_light_callback,
                                            10)
        self.sub_green_light = self.create_subscription(Float32, 
                                            'traffic_light/green/density', 
                                            self.green_light_callback,
                                            10)
        
        self.msg_img_prop_frame = Image()
        self.msg_control = Twist()

        self.timer_period = 0.1

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        ###### variables ######

        self.bridge_object = CvBridge()
        self.previous_d = 0
        self.k = 0.00015
        self.v_linear = 0.06
        self.kernel_erode = np.ones((5,5), np.uint8)
        self.kernel_dilated = np.ones((7,7),np.uint8)

        # densities ##
        self.green_density = 0.0
        self.red_density = 0.0

        self.traffic_flag = False
        self.hor_flag = False
        
        
        ###### logger ######
        self.get_logger().info('mira mama sigo lineas!')

    def green_light_callback(self,msg):
        self.green_density = msg.data 
        if self.green_density > 0.15:
            self.traffic_flag = True
            self.hor_flag = False

    
    def red_light_callback(self,msg):
        self.red_density = msg.data
        if self.red_density > 0.15:
            self.traffic_flag = False
    
    
    def line_detection_callback(self, msg):

        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0 

        img = self.bridge_object.imgmsg_to_cv2(msg, 'rgb8')
        img_rotated = cv2.rotate(img, cv2.ROTATE_180)

        M = np.ones(img_rotated.shape, dtype='uint8')*60
        img_added = cv2.subtract(img_rotated, M)

        img_cropped = img_added[int((img_rotated.shape[0])*(3/4)):int((img_rotated.shape[0])), 1:int((img_rotated.shape[1]))]
        img_blurred = cv2.GaussianBlur(img_cropped, (5,5),0)

        ### ref ###
        
        img_canoso = cv2.Canny(img_blurred,20,30)
        img_canny = cv2.morphologyEx(img_canoso, cv2.MORPH_CLOSE, kernel=self.kernel_erode)
        # img_dilated = cv2.dilate(img_canoso, self.kernel_dilated, 2)
        # img_canny = cv2.erode(img_dilated, self.kernel_erode, 1)

        x1_ref = 0 
        y1_ref = int(img_canny.shape[1]/2)

        x2_ref = int(img_canny.shape[0])
        y2_ref = int(img_canny.shape[1]/2)

        lines = cv2.HoughLinesP(img_canny,1,np.pi/180,10,10,20)
        i_max = 0

        if self.traffic_flag == True and self.hor_flag == False:
            if lines is not None: 
                self.hor_flag = False
                for x in range(0,len(lines)):
                    for x1,y1,x2,y2 in lines[x]:
                        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        if self.previous_d < distance:
                            i_max = x
                        self.previous_d = distance


                x1 = lines[i_max][0][0]
                y1 = lines[i_max][0][1]
                x2 = lines[i_max][0][2]
                y2 = lines[i_max][0][3]

                cv2.line(img_canny,(x1,y1),(x2,y2),(255,0,0),3)
            else:
                print("No hay lines :(")

            cv2.line(img_canny,(y1_ref,x1_ref),(y2_ref,x2_ref),(255,0,0),3)

            ang = np.arctan((y2-y1) / ((x2 - x1)+0.00001))
            delta_y = np.abs(y2 - y1)
            delta_x = np.abs(x2 - x1)
            print(delta_y, " ", delta_x)
            
            if (delta_y > 0 and delta_y < 3) and (delta_x > 40):
                print("lin_hor")
                self.hor_flag = True
                self.traffic_flag = False
            
            error = x2 - y2_ref
            self.v_ang = -(self.k * error)
            self.v_linear = 0.06


        else:
            self.v_linear = 0.0
            self.v_ang = 0.0

        #print(self.v_ang)
    
        ##msg assign
        self.msg_control.angular.z = self.v_ang
        self.msg_control.linear.x = self.v_linear

        self.msg_img_prop_frame = self.bridge_object.cv2_to_imgmsg(img_canny)

    def timer_callback(self):
        self.publisher_img_prop_frame.publish(self.msg_img_prop_frame)
        self.publisher_cmd_vel.publish(self.msg_control)
        
        
        
def main(args = None):
    rclpy.init(args=args)
    line_detection = line_detection_node()
    rclpy.spin(line_detection) #While(1)
    line_detection.destroy_node() #Destruir nodo
    rclpy.shutdown()    

if __name__ == '__main__':
    main()