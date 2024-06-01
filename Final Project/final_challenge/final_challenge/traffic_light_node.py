import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
import rclpy.qos
import cv2 
from cv_bridge import CvBridge


class traffic_light_node(Node):

    def __init__(self):
        super().__init__('traffic_light_node')

        self.bridge_object = CvBridge()
        
        self.publisher_img_prop_rMask = self.create_publisher(Image, 'traffic_light/red/mask', 10)
        self.publisher_img_prop_rDens = self.create_publisher(Float32, 'traffic_light/red/density', 10)
        
        self.publisher_img_prop_gMask = self.create_publisher(Image, 'traffic_light/green/mask', 10)
        self.publisher_img_prop_gDens = self.create_publisher(Float32, 'traffic_light/green/density', 10)

        self.publisher_img_prop_yMask = self.create_publisher(Image, 'img_properties/yellow/mask', 10)
        self.publisher_img_prop_yDens = self.create_publisher(Float32, 'img_properties/yellow/density', 10)
    

        self.sub_video = self.create_subscription(Image, 
                                            'video_source/raw', 
                                            self.frame_treat_callback,
                                            10)

        self.msg_img_prop_rMask = Image()
        self.msg_img_prop_rDens = Float32()
        
        self.msg_img_prop_gMask = Image()
        self.msg_img_prop_gDens = Float32()

        self.msg_img_prop_yMask = Image()
        self.msg_img_prop_yDens = Float32()
    

        #Density
        self.r_density = 0.0
        self.g_density = 0.0
        self.y_density = 0.0

        self.timer_period = 0.05

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.k = np.ones((11,11), np.uint8)

        self.get_logger().info('SoI DalTonIcO xd!')

    
    def frame_treat_callback(self, msg):
        #Rotate image and convert to hsv

        #self.get_logger().info('Read image')
        print("Read image")
        img = self.bridge_object.imgmsg_to_cv2(msg, 'rgb8')
        img_rotated = cv2.rotate(img, cv2.ROTATE_180)
        img_hsv = cv2.cvtColor(img_rotated, cv2.COLOR_RGB2HSV)
        

        ################  RED ################
        ## mask definition ##

        lo_mask_r = img_hsv[:,:,0] > 150.0
        hi_mask_r = img_hsv[:,:,0] < 200.0

        lo_mask_r_2 = img_hsv[:,:,0] > 0.0 
        hi_mask_r_2 = img_hsv[:,:,0] < 10.0

        r_full_mask = (lo_mask_r * hi_mask_r) + (lo_mask_r_2 * hi_mask_r_2)

        saturation_mask_r = img_hsv[:,:,1] > 120.0

        mask_r = r_full_mask*saturation_mask_r

        r_red = img_rotated[:,:,0]*mask_r
        r_green = img_rotated[:,:,1]*mask_r
        r_blue = img_rotated[:,:,2]*mask_r

        img_masked_r = np.dstack((r_red, r_green, r_blue))
        img_gray_r = cv2.cvtColor(img_masked_r, cv2.COLOR_RGB2GRAY)
        r_median = cv2.medianBlur(img_gray_r,7)
        _, img_th_r = cv2.threshold(r_median, 1, 255, cv2.THRESH_BINARY)
    
        img_dil_r = cv2.dilate(img_th_r, self.k, 1)
        img_ero_r = cv2.erode(img_dil_r, self.k, 2)
        self.msg_img_prop_rMask = self.bridge_object.cv2_to_imgmsg(img_ero_r)

        #Density
        r_total_pix = img_ero_r.shape[0] * img_ero_r.shape[1]
        r_num_max_pix = cv2.countNonZero(img_ero_r)
        self.r_density = r_num_max_pix/r_total_pix
        

        ################  GREEN ################
        ## green mask ##

        lo_mask_g = img_hsv[:,:,0] > 35.0
        hi_mask_g = img_hsv[:,:,0] < 100.0

        saturation_mask_g = img_hsv[:,:,1] > 100.0

        mask_g = lo_mask_g*hi_mask_g*saturation_mask_g
        g_red = img_rotated[:,:,0]*mask_g
        g_green = img_rotated[:,:,1]*mask_g
        g_blue = img_rotated[:,:,2]*mask_g

        img_masked_g = np.dstack((g_red, g_green, g_blue))
        img_gray_g = cv2.cvtColor(img_masked_g, cv2.COLOR_RGB2GRAY)
        g_median = cv2.medianBlur(img_gray_g,7)
        _, img_th_g = cv2.threshold(g_median, 1, 255, cv2.THRESH_BINARY)
        
        img_dil_g = cv2.dilate(img_th_g, self.k, 1)
        img_ero_g = cv2.erode(img_dil_g, self.k, 2)
        self.msg_img_prop_gMask = self.bridge_object.cv2_to_imgmsg(img_ero_g)

        #Density
        g_total_pix = img_ero_g.shape[0] * img_ero_g.shape[1]
        g_num_max_pix = cv2.countNonZero(img_ero_g)
        self.g_density = g_num_max_pix/g_total_pix


        ################  YELLOW ################
        ## yellow mask ##

        lo_mask_y = img_hsv[:,:,0] > 20.0
        hi_mask_y = img_hsv[:,:,0] < 35.0

        saturation_mask_y = img_hsv[:,:,1] > 100.0

        mask_y = lo_mask_y*hi_mask_y*saturation_mask_y
        y_red = img_rotated[:,:,0]*mask_y
        y_green = img_rotated[:,:,1]*mask_y
        y_blue = img_rotated[:,:,2]*mask_y

        img_masked_y = np.dstack((y_red, y_green, y_blue))

        img_gray_y = cv2.cvtColor(img_masked_y, cv2.COLOR_RGB2GRAY)
        y_median = cv2.medianBlur(img_gray_y,7)
        _, img_th_y = cv2.threshold(y_median, 1, 255, cv2.THRESH_BINARY)
        
        img_dil_y = cv2.dilate(img_th_y, self.k, 1)
        img_ero_y = cv2.erode(img_dil_y, self.k, 2)
        self.msg_img_prop_yMask = self.bridge_object.cv2_to_imgmsg(img_ero_y)

        #Density
        y_total_pix = img_ero_y.shape[0] * img_ero_y.shape[1]
        y_num_max_pix = cv2.countNonZero(img_ero_y)
        self.y_density = y_num_max_pix/y_total_pix


        
    def timer_callback(self):
        #message assignment
        #RED messages
        self.msg_img_prop_rDens.data = self.r_density

        #GREEN messages
        self.msg_img_prop_gDens.data = self.g_density

        #YELOW messages
        self.msg_img_prop_yDens.data = self.y_density

        #Publish messages
        self.publisher_img_prop_rMask.publish(self.msg_img_prop_rMask)
        self.publisher_img_prop_rDens.publish(self.msg_img_prop_rDens)
       
        self.publisher_img_prop_gMask.publish(self.msg_img_prop_gMask)
        self.publisher_img_prop_gDens.publish(self.msg_img_prop_gDens)

        self.publisher_img_prop_yMask.publish(self.msg_img_prop_yMask)
        self.publisher_img_prop_yDens.publish(self.msg_img_prop_yDens)
        

       
        
def main(args = None):
    rclpy.init(args=args)
    traffic_light = traffic_light_node()
    rclpy.spin(traffic_light) #While(1)
    traffic_light.destroy_node() #Destruir nodo
    rclpy.shutdown()    

if __name__ == '__main__':
    main()