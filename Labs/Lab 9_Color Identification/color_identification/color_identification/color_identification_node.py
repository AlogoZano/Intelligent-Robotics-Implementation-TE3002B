import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
import rclpy.qos
import cv2 
from cv_bridge import CvBridge


class color_identification_node(Node):

    def __init__(self):
        super().__init__('color_identification_node')

        self.bridge_object = CvBridge()
        
        self.publisher_img_prop_rMask = self.create_publisher(Image, 'img_properties/red/mask', 10)
        self.publisher_img_prop_rDens = self.create_publisher(Float32, 'img_properties/red/density', 10)
        self.publisher_img_prop_rXY = self.create_publisher(Point, 'img_properties/red/xy', 10)

        self.publisher_img_prop_gMask = self.create_publisher(Image, 'img_properties/green/mask', 10)
        self.publisher_img_prop_gDens = self.create_publisher(Float32, 'img_properties/green/density', 10)
        self.publisher_img_prop_gXY = self.create_publisher(Point, 'img_properties/green/xy', 10)

        self.publisher_img_prop_yMask = self.create_publisher(Image, 'img_properties/yellow/mask', 10)
        self.publisher_img_prop_yDens = self.create_publisher(Float32, 'img_properties/yellow/density', 10)
        self.publisher_img_prop_yXY = self.create_publisher(Point, 'img_properties/yellow/xy', 10)

        self.publisher_img_prop_bMask = self.create_publisher(Image, 'img_properties/blue/mask', 10)
        self.publisher_img_prop_bDens = self.create_publisher(Float32, 'img_properties/blue/density', 10)
        self.publisher_img_prop_bXY = self.create_publisher(Point, 'img_properties/blue/xy', 10)

        self.publisher_img_prop_wMask = self.create_publisher(Image, 'img_properties/white/mask', 10)
        self.publisher_img_prop_wDens = self.create_publisher(Float32, 'img_properties/white/density', 10)
        self.publisher_img_prop_wXY = self.create_publisher(Point, 'img_properties/white/xy', 10)

        self.sub_video = self.create_subscription(Image, 
                                            'video_source/raw', 
                                            self.frame_treat_callback,
                                            10)

        self.msg_img_prop_rMask = Image()
        self.msg_img_prop_rDens = Float32()
        self.msg_img_prop_rXY = Point()

        self.msg_img_prop_gMask = Image()
        self.msg_img_prop_gDens = Float32()
        self.msg_img_prop_gXY = Point()

        self.msg_img_prop_yMask = Image()
        self.msg_img_prop_yDens = Float32()
        self.msg_img_prop_yXY = Point()

        self.msg_img_prop_bMask = Image()
        self.msg_img_prop_bDens = Float32()
        self.msg_img_prop_bXY = Point()

        self.msg_img_prop_wMask = Image()
        self.msg_img_prop_wDens = Float32()
        self.msg_img_prop_wXY = Point()

        #Density
        self.r_density = 0.0
        self.g_density = 0.0
        self.b_density = 0.0
        self.y_density = 0.0
        self.w_density = 0.0

        #Center of mass
        self.r_cX = 0.0
        self.r_cY = 0.0

        self.g_cX = 0.0
        self.g_cY = 0.0

        self.b_cX = 0.0
        self.b_cY = 0.0

        self.y_cX = 0.0
        self.y_cY = 0.0

        self.w_cX = 0.0
        self.w_cY = 0.0

        self.timer_period = 0.1

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.k = np.ones((5,5), np.uint8)
        self.choice = 4
        self.get_logger().info('SoI DalTonIcO xd!')

    
    def frame_treat_callback(self, msg):
        #Rotate image and convert to hsv

        #self.get_logger().info('Read image')
        print("Read image")
        img = self.bridge_object.imgmsg_to_cv2(msg, 'rgb8')
        img_flipped = cv2.flip(img, 1)
        img_rotated = cv2.rotate(img_flipped, cv2.ROTATE_180)
        img_hsv = cv2.cvtColor(img_rotated, cv2.COLOR_RGB2HSV)
        
############################################################################################

        #RED MASK
        if(self.choice == 0):
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
            
            #Center of mass
            r_M = cv2.moments(img_ero_r)
            self.r_cX = (r_M["m10"]/(r_M["m00"]+0.001))
            self.r_cY = (r_M["m01"]/(r_M["m00"]+0.001))

        ################################################################################3

        #GREEN MASK
        elif(self.choice == 1):
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
            
            #Center of mass
            g_M = cv2.moments(img_ero_g)
            self.g_cX = (g_M["m10"]/(g_M["m00"]+0.001))
            self.g_cY = (g_M["m01"]/(g_M["m00"]+0.001))

        ################################################################################

        #YELLOW MASK

        ##first option mask, change *img_masked_y* if fisrt option doesnt work

        elif(self.choice == 2):
            lo_mask_y = img_hsv[:,:,0] > 20.0
            hi_mask_y = img_hsv[:,:,0] < 55.0

            saturation_mask_y = img_hsv[:,:,1] > 80.0


            mask_y = lo_mask_y*hi_mask_y*saturation_mask_y
            y_red = img_rotated[:,:,0]*mask_y
            y_green = img_rotated[:,:,1]*mask_y
            y_blue = img_rotated[:,:,2]*mask_y

            img_masked_y = np.dstack((y_red, y_green, y_blue))

            ##second option mask, change *img_2masked_y* if fisrt option doesnt work

            # light_y = [40,255,255]
            # dark_y = [25,130,120]

            # img_2masked_y = cv2.inRange(img_hsv, dark_y, light_y)

            ##

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
            
            #Center of mass
            y_M = cv2.moments(img_ero_y)
            self.y_cX = (y_M["m10"]/(y_M["m00"]+0.001))
            self.y_cY = (y_M["m01"]/(y_M["m00"]+0.001))
        
        ################################################################################

        #BLUE MASK
        elif(self.choice == 3):
            lo_mask_b = img_hsv[:,:,0] > 100.0
            hi_mask_b = img_hsv[:,:,0] < 140.0

            saturation_mask_b = img_hsv[:,:,1] > 100.0

            mask_b = lo_mask_b*hi_mask_b*saturation_mask_b
            b_red = img_rotated[:,:,0]*mask_b
            b_green = img_rotated[:,:,1]*mask_b
            b_blue = img_rotated[:,:,2]*mask_b

            img_masked_b = np.dstack((b_red, b_green, b_blue))
            img_gray_b = cv2.cvtColor(img_masked_b, cv2.COLOR_RGB2GRAY)
            b_median = cv2.medianBlur(img_gray_b,7)
            _, img_th_b = cv2.threshold(b_median, 1, 255, cv2.THRESH_BINARY)
            
            img_dil_b = cv2.dilate(img_th_b, self.k, 1)
            img_ero_b = cv2.erode(img_dil_b, self.k, 2)
            self.msg_img_prop_bMask = self.bridge_object.cv2_to_imgmsg(img_ero_b)

            #Density
            b_total_pix = img_ero_b.shape[0] * img_ero_b.shape[1]
            b_num_max_pix = cv2.countNonZero(img_ero_b)
            self.b_density = b_num_max_pix/b_total_pix
            
            #Center of mass
            b_M = cv2.moments(img_ero_b)
            self.b_cX = (b_M["m10"]/(b_M["m00"]+0.001))
            self.b_cY = (b_M["m01"]/(b_M["m00"]+0.001))

        ################################################################################

        #WHITE MASK
        elif(self.choice == 4):

            saturation_lo_mask_w = img_hsv[:,:,1] > 0.0
            saturation_hi_mask_w = img_hsv[:,:,1] < 40.0
            hue_mask_w_h = img_hsv[:,:,0] < 255.0
            hue_mask_w_l = img_hsv[:,:,0] > 0.0

            mask_w = saturation_lo_mask_w*saturation_hi_mask_w*hue_mask_w_l*hue_mask_w_h
            w_red = img_rotated[:,:,0]*mask_w
            w_green = img_rotated[:,:,1]*mask_w
            w_blue = img_rotated[:,:,2]*mask_w

            img_masked_w = np.dstack((w_red, w_green, w_blue))
            img_gray_w = cv2.cvtColor(img_masked_w, cv2.COLOR_RGB2GRAY)
            w_median = cv2.medianBlur(img_gray_w,7)
            _, img_th_w = cv2.threshold(w_median, 1, 255, cv2.THRESH_BINARY)
            
            img_dil_w = cv2.dilate(img_th_w, self.k, 1)
            img_ero_w = cv2.erode(img_dil_w, self.k, 2)
            self.msg_img_prop_wMask = self.bridge_object.cv2_to_imgmsg(img_ero_w)

            #Density
            w_total_pix = img_ero_w.shape[0] * img_ero_w.shape[1]
            w_num_max_pix = cv2.countNonZero(img_ero_w)
            self.w_density = w_num_max_pix/w_total_pix
            
            #Center of mass
            w_M = cv2.moments(img_ero_w)
            self.w_cX = (w_M["m10"]/(w_M["m00"]+0.001))
            self.w_cY = (w_M["m01"]/(w_M["m00"]+0.001))

    
    def timer_callback(self):
        #message assignment
        #RED messages
        self.msg_img_prop_rDens.data = self.r_density
        self.msg_img_prop_rXY.x = self.r_cX
        self.msg_img_prop_rXY.y = self.r_cY
        #GREEN messages
        self.msg_img_prop_gDens.data = self.g_density
        self.msg_img_prop_gXY.x = self.g_cX
        self.msg_img_prop_gXY.y = self.g_cY
        #YELLOW messages
        self.msg_img_prop_yDens.data = self.y_density
        self.msg_img_prop_yXY.x = self.y_cX
        self.msg_img_prop_yXY.y = self.y_cY
        #BLUE messages
        self.msg_img_prop_bDens.data = self.b_density
        self.msg_img_prop_bXY.x = self.b_cX
        self.msg_img_prop_bXY.y = self.b_cY
        #WHITE messages
        self.msg_img_prop_wDens.data = self.w_density
        self.msg_img_prop_wXY.x = self.w_cX
        self.msg_img_prop_wXY.y = self.w_cY

        #Publish messages
        self.publisher_img_prop_rMask.publish(self.msg_img_prop_rMask)
        self.publisher_img_prop_rDens.publish(self.msg_img_prop_rDens)
        self.publisher_img_prop_rXY.publish(self.msg_img_prop_rXY)

        self.publisher_img_prop_gMask.publish(self.msg_img_prop_gMask)
        self.publisher_img_prop_gDens.publish(self.msg_img_prop_gDens)
        self.publisher_img_prop_gXY.publish(self.msg_img_prop_gXY)

        self.publisher_img_prop_yMask.publish(self.msg_img_prop_yMask)
        self.publisher_img_prop_yDens.publish(self.msg_img_prop_yDens)
        self.publisher_img_prop_yXY.publish(self.msg_img_prop_yXY)

        self.publisher_img_prop_bMask.publish(self.msg_img_prop_bMask)
        self.publisher_img_prop_bDens.publish(self.msg_img_prop_bDens)
        self.publisher_img_prop_bXY.publish(self.msg_img_prop_bXY)

        self.publisher_img_prop_wMask.publish(self.msg_img_prop_wMask)
        self.publisher_img_prop_wDens.publish(self.msg_img_prop_wDens)
        self.publisher_img_prop_wXY.publish(self.msg_img_prop_wXY)
        
        
def main(args = None):
    rclpy.init(args=args)
    color_identification = color_identification_node()
    rclpy.spin(color_identification) #While(1)
    color_identification.destroy_node() #Destruir nodo
    rclpy.shutdown()    

if __name__ == '__main__':
    main()
