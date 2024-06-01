import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
import rclpy.qos
import cv2 
from cv_bridge import CvBridge


class img_process_node(Node):

    def __init__(self):
        super().__init__('image_processing_node')
        

        self.publisher_img_YOLO = self.create_publisher(Image, 'img/yolo', 10)
        self.publisher_img_PATH = self.create_publisher(Image, 'img/path', 10)
        self.publisher_img_CROSS = self.create_publisher(Image, 'img/crosswalk', 10)


       
        self.sub_video = self.create_subscription(Image, 
                                            'video_source/raw', 
                                            self.camera_callback,
                                            10)
        
    
        self.msg_img_yolo = Image()
        self.msg_img_path = Image()
        self.msg_img_crosswalk = Image()
        
        ##timer
        self.timer_period = 0.2
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        ###### variables ######
        self.bridge_object = CvBridge()
        self.kernel_erode = np.ones((9,9), np.uint8)
        self.scale = 0.1
    
        
        ###### logger ######
        self.get_logger().info('Proceso imágenes :)')

    
    def camera_callback(self, msg):
        img = self.bridge_object.imgmsg_to_cv2(msg, 'rgb8')
        if img is not None:
            img_rotated = cv2.rotate(img, cv2.ROTATE_180)

            #Process for path detection
            img_cropped_path = img_rotated[int((img_rotated.shape[0])*(8/9)):int((img_rotated.shape[0])), int((img_rotated.shape[1])*2/9):int((img_rotated.shape[1])*7/9)]

            M = np.ones(img_cropped_path.shape, dtype='uint8')*120
            img_added_path = cv2.add(img_cropped_path, M)
            img_blurred_path = cv2.GaussianBlur(img_added_path, (5,5),0)     
            img_canny_path = cv2.Canny(img_blurred_path,20,30)
            img_path = cv2.morphologyEx(img_canny_path, cv2.MORPH_CLOSE, kernel=self.kernel_erode)
            self.msg_img_path = self.bridge_object.cv2_to_imgmsg(img_path, 'mono8')

            #Process for yolo testing
            img_cropped_yolo = img_rotated[int((img_rotated.shape[0])*(3/10)):int((img_rotated.shape[0])*(6/10)), int((img_rotated.shape[1])*2/9):int((img_rotated.shape[1])*7/9)]
            img_reduced_yolo = cv2.resize(img_cropped_yolo, None, fx = self.scale, fy = self.scale, interpolation=cv2.INTER_AREA)
            self.msg_img_yolo = self.bridge_object.cv2_to_imgmsg(img_reduced_yolo, 'rgb8')
            print(img_reduced_yolo.shape)

            #Process for crosswalk detection
            img_cropped_crosswalk = img_rotated[int((img_rotated.shape[0])*(8/9)):int((img_rotated.shape[0])), int((img_rotated.shape[1])*2/9):int((img_rotated.shape[1])*7/9)]
            c_gray = cv2.cvtColor(img_cropped_crosswalk, cv2.COLOR_RGB2GRAY)
            c_invert = cv2.bitwise_not(c_gray)
            c_blurred = cv2.GaussianBlur(c_invert, (25,25), 0)
            _, c_thresh = cv2.threshold(c_blurred, 127, 255, cv2.THRESH_BINARY)
            self.msg_img_crosswalk = self.bridge_object.cv2_to_imgmsg(c_thresh, 'mono8')



    def timer_callback(self):
        #publish messages#
        self.publisher_img_PATH.publish(self.msg_img_path)
        self.publisher_img_YOLO.publish(self.msg_img_yolo)
        self.publisher_img_CROSS.publish(self.msg_img_crosswalk)
        
        
def main(args = None):
    rclpy.init(args=args)
    img_pro = img_process_node()
    rclpy.spin(img_pro) #While(1)
    img_pro.destroy_node() #Destruir nodo
    rclpy.shutdown()    

if __name__ == '__main__':
    main()