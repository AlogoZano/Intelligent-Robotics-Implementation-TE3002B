import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import Image
import rclpy.qos
import cv2 
from cv_bridge import CvBridge


class line_detection_node(Node):

    def __init__(self):
        super().__init__('line_detection_node')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('brightness', rclpy.Parameter.Type.DOUBLE),
            ]
        )

        self.publisher_img_prop_lines = self.create_publisher(Image, 'img_properties/frame/lines', 10)
        self.publisher_img_prop_crosswalk = self.create_publisher(Image, 'img_properties/frame/crosswalk', 10)
        self.publisher_line_error = self.create_publisher(Integer, 'line_error', 10)
        self.publisher_cross_walk = self.create_publisher(Bool, 'cross_walk_flag', 10)
       
        self.sub_video = self.create_subscription(Image, 
                                            'video_source/raw', 
                                            self.line_detection_callback,
                                            10)
    
        
        self.msg_img_prop_lines = Image()
        self.msg_img_prop_crosswalk = Image()
        self.msg_lines_error = Integer()
        self.msg_crossWalk_flag = Bool()
        
        ##timer
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        ###### variables ######
        self.bridge_object = CvBridge()
        self.previous_d = 0
        self.error = 0
        
        self.kernel_erode = np.ones((5,5), np.uint8)
        
        # densities ##
        self.green_density = 0.0
        self.red_density = 0.0
        
        # flags #
        self.crossWalk_flag = False
        
        ###### logger ######
        self.get_logger().info('JESUU veo Lineas!')
    
    def line_detection_callback(self, msg):

        #initializing the points of the lines
        x1,x2,y1,y2 = 0
        rect_count, sqr_count= 0
        i_max = 0
        
        img = self.bridge_object.imgmsg_to_cv2(msg, 'rgb8')
        img_rotated = cv2.rotate(img, cv2.ROTATE_180)
        img_cropped = img_rotated[int((img_rotated.shape[0])*(6/7)):int((img_rotated.shape[0])), int((img_rotated.shape[1])*2/9):int((img_rotated.shape[1])*7/9)]
        
        M = np.ones(img_cropped.shape, dtype='uint8')*self.get_parameter('brightness').get_parameter_value().double_value
        img_added = cv2.add(img_cropped, M)
        
        # line detection method
        img_blurred = cv2.GaussianBlur(img_added, (5,5),0)     
        img_canny = cv2.Canny(img_blurred,20,30)
        img_lines = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel=self.kernel_erode)
       
        x1_ref = 0 
        y1_ref = int(img_lines.shape[1]/2)
        x2_ref = int(img_lines.shape[0])
        y2_ref = int(img_lines.shape[1]/2)
        
        cv2.line(img_lines,(y1_ref,x1_ref),(y2_ref,x2_ref),(255,0,0),3)

        lines = cv2.HoughLinesP(img_lines,1,np.pi/180,10,10,20)
            
        if lines is not None: 
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
                        
            self.error = x2 - y2_ref
            
        else:
            print("No hay lines :(")
            self.error = self.error
        #   
        
        # new cross walk detection method
        blur2 = cv2.GaussianBlur(img_added, (25,25),0)
        ret,thresh = cv2.threshold(blur2,180,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
        
        for cnt in contours:
            x3,y3 = cnt[0][0]
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x4, y5, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if ratio >= 0.9 and ratio <= 1.1:
                frame = cv2.drawContours(frame, [cnt], -1, (0,255,255), 3)
                cv2.putText(frame, 'Square', (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                sqr_count += 1
            else:
                cv2.putText(frame, 'Rectangle', (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                frame = cv2.drawContours(frame, [cnt], -1, (0,255,0), 3)
                rect_count += 1
            self.msg_img_prop_crosswalk = self.bridge_object.cv2_to_imgmsg(frame)
            
        if rect_count > 2 or sqr_count > 2:
            self.crossWalk_flag = True
        else:
            self.crossWalk_flag = False
        # 
        
        self.msg_img_prop_crosswalk = self.bridge_object.cv2_to_imgmsg(img_lines)
        self.msg_lines_error = self.error
        self.msg_crossWalk_flag = self.crossWalk_flag


    def timer_callback(self):
        #publish messages#
        self.publisher_img_prop_crosswalk.publish(self.msg_img_prop_crosswalk)
        self.publisher_img_prop_lines.publish(self.msg_img_prop_lines)
        self.publisher_cross_walk.publish(self.msg_crossWalk_flag)
        self.publisher_line_error.publish(self.msg_lines_error)
        
        
        
def main(args = None):
    rclpy.init(args=args)
    line_detection = line_detection_node()
    rclpy.spin(line_detection) #While(1)
    line_detection.destroy_node() #Destruir nodo
    rclpy.shutdown()    

if __name__ == '__main__':
    main()