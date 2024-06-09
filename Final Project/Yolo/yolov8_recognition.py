
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
from collections import deque, Counter
import cv2

from yolo_msg.msg import InferenceResult
from yolo_msg.msg import Yolov8Inference

from cv_bridge import CvBridgeError

class YoloInference(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.model = YOLO('/home/isra/ros2_ws/src/yolov8_ros2/yolov8_ros2/last.pt')
        self.yolo_msg = Yolov8Inference()
        self.img = np.ndarray((72, 128, 3))
        
        self.bridge = CvBridge()
        self.valid_img = False  # Inicialización de valid_img

        self.sub = self.create_subscription(Image, 'img/yolo', self.camera_callback, 10)
        self.yolo_pub = self.create_publisher(Yolov8Inference, '/Yolov8_inference', 10)
        self.publisher_signal_class = self.create_publisher(Int8, '/class_num', 10)
        self.yolo_img_pub = self.create_publisher(Image, '/inference_result', 10)

        self.signal_class_msg = Int8()
        
        self.buffer = deque(maxlen=30)
        self.moda = 0
        self.prev_moda = -1

        timer_period = 0.2
        # timer_period2 = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.timer_2 = self.create_timer(timer_period2, self.timer_callback2)

    def camera_callback(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.valid_img = True
        except CvBridgeError as e:
            self.get_logger().info(f'Failed to get an image: {e}')
            self.valid_img = False

    def timer_callback(self):
        if self.valid_img:
            results = self.model(self.img)
            self.yolo_msg.header.frame_id = 'inference'
            self.yolo_msg.header.stamp = self.get_clock().now().to_msg()

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    self.inference_result = InferenceResult()
                    # b = box.xyxy[0].to('cpu').detach().numpy().copy()
                    c = box.cls
                    self.inference_result.class_name = self.model.names[int(c)]
                    # self.inference_result.top = int(b[0])
                    # self.inference_result.left = int(b[1])
                    # self.inference_result.bottom = int(b[2])
                    # self.inference_result.right = int(b[3])
                    self.yolo_msg.yolov8_inference.append(self.inference_result)
                    
            
            #frame = results[0].plot()
            
            # Convert float64 image to uint8
            #frame_uint8 = (frame * 255).astype(np.uint8)
            
            # self.yolo_pub.publish(self.yolo_msg)

                    if self.inference_result.class_name == "straight":
                        self.num= 1
                    elif self.inference_result.class_name == "turnleft":
                        self.num = 2
                    elif self.inference_result.class_name == "turnright":
                        self.num = 3
                    elif self.inference_result.class_name == "work":
                        self.num = 4
                    elif self.inference_result.class_name == "giveway":
                        self.num = 5
                    elif self.inference_result.class_name == "stop":
                        self.num = 6
                    elif self.inference_result.class_name == '':
                        self.num = 0                    

                    self.buffer.append(self.num)
                    print(self.moda)

        if (len(self.buffer)) == self.buffer.maxlen:
            self.moda = Counter(self.buffer).most_common(1)[0][0]
            self.signal_class_msg.data = self.moda

            if (self.prev_moda != self.moda):
                self.publisher_signal_class.publish(self.signal_class_msg)
            
            self.prev_moda = self.moda
                
            self.yolo_msg.yolov8_inference.clear()
            
          

    # def timer_callback2(self):
    #     if self.valid_img:
    #         self.yolo_img_pub.publish(self.bridge.cv2_to_imgmsg(self.img, encoding='bgr8'))

            

def main(args=None):
    rclpy.init(args=args)
    y_i = YoloInference()
    rclpy.spin(y_i)
    y_i.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

   
