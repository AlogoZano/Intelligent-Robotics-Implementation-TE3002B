import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, Bool, String, Int8
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import Image
import rclpy.qos
import cv2 
from cv_bridge import CvBridge


class path_detection_node(Node):

    def __init__(self):
        super().__init__('path_detection_node')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('brightness', rclpy.Parameter.Type.DOUBLE),
            ]
        )

        self.publisher_img_prop_lines = self.create_publisher(Image, 'img_properties/frame/lines', 10)
        self.publisher_img_reduced = self.create_publisher(Image, 'img_properties/frame/reduced', 10)
        self.publisher_img_prop_crosswalk = self.create_publisher(Image, 'img_properties/frame/crosswalk', 10)
        self.publisher_line_error = self.create_publisher(Int32, 'line_error', 10)
        self.publisher_cross_walk = self.create_publisher(Bool, 'cross_walk_flag', 10)
       
        self.sub_video = self.create_subscription(Image, 
                                            'img/path', 
                                            self.line_detection_callback,
                                            10)
        
        self.sub_video = self.create_subscription(Image, 
                                    'img/crosswalk', 
                                    self.crosswalk_detection_callback,
                                    10)
        
        
    
        
        self.msg_img_prop_lines = Image()
        self.msg_img_prop_crosswalk = Image()
        self.msg_lines_error = Int32()
        self.msg_crossWalk_flag = Bool()
        self.msg_signal = Int8()
        
        ##timer
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        ###### variables ######
        self.bridge_object = CvBridge()
        self.previous_d = 0
        self.error = 0
        
        self.kernel_erode = np.ones((9,9), np.uint8)
        
        # densities ##
        self.green_density = 0.0
        self.red_density = 0.0
        
        # flags #
        self.crossWalk_flag = True

        
        ###### logger ######
        self.get_logger().info('JESUU veo Lineas!')
    
    def line_detection_callback(self, msg):

        #initializing the points of the lines
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0
        rect_count = 0
        sqr_count= 0
        i_max = 0
        approx = []
        
        img_lines = self.bridge_object.imgmsg_to_cv2(msg, 'mono8')

       
        x1_ref = 0 
        y1_ref = int(img_lines.shape[1]/2)
        x2_ref = int(img_lines.shape[0])
        y2_ref = int(img_lines.shape[1]/2)
        
        cv2.line(img_lines,(y1_ref,x1_ref),(y2_ref,x2_ref),(255,0,0),3)

        lines = cv2.HoughLinesP(img_lines,1,np.pi/180,10,10,20)
            
        if lines is not None: 
            x1_sum,y1_sum,x2_sum,y2_sum = 0,0,0,0
            num_lines = len(lines) 
            self.hor_flag = False
            
            for line in lines:
                for x1,y1,x2,y2 in line:
                    x1_sum += x1
                    y1_sum += y1
                    x2_sum += x2
                    y2_sum += y2

            x1_avg = x1_sum//num_lines
            x2_avg = x2_sum//num_lines
            y1_avg = y1_sum//num_lines
            y2_avg = y2_sum//num_lines
            

            cv2.line(img_lines,(x1_avg,y1_avg),(x2_avg,y2_avg),(255,0,0),3)

            self.error = x2_avg - y2_ref
            print("error: ", self.error)
            
        else:
            print("No hay lines :(")
            self.error = self.error
        #   
        
        self.msg_img_prop_lines = self.bridge_object.cv2_to_imgmsg(img_lines)
        self.msg_lines_error.data = int(self.error)
        self.msg_crossWalk_flag.data = self.crossWalk_flag

    def crosswalk_detection_callback(self, msg):
        thresh = self.bridge_object.imgmsg_to_cv2(msg, 'mono8')
        contours, _ = cv2.findContours(thresh, 1, 2)
        crosswalk_value = 0
        sqr_count = 0
        rect_count = 0

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

            if len(approx) > 4 and len(approx) < 8:
                _, _, w, h = cv2.boundingRect(cnt)
                ratio = float(w)/h
                if ratio >= 0.9 and ratio <= 1.1:
                    sqr_count += 1
                else:
                    rect_count += 1
                    
        crosswalk_value = sqr_count + rect_count
            
        if crosswalk_value >= 3:
            self.crossWalk_flag = True
        else:
            self.crossWalk_flag = False

        self.msg_crossWalk_flag.data = self.crossWalk_flag
            

    def timer_callback(self):
        #publish messages#
        self.publisher_img_prop_crosswalk.publish(self.msg_img_prop_crosswalk)
        self.publisher_img_prop_lines.publish(self.msg_img_prop_lines)
        self.publisher_cross_walk.publish(self.msg_crossWalk_flag)
        self.publisher_line_error.publish(self.msg_lines_error)

        
        
        
def main(args = None):
    rclpy.init(args=args)
    path_detection = path_detection_node()
    rclpy.spin(path_detection) #While(1)
    path_detection.destroy_node() #Destruir nodo
    rclpy.shutdown()    

if __name__ == '__main__':
    main()