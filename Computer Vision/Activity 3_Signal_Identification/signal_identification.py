#Adrián Lozano González
#Israel Macías Santana
#B.Sc. in Robotics and Digital Systems Enginering

############################################################################################################################

#TRAFFIC SIGNS IDENTIFICATION

#Further details are contained within the jupyter notebook (signal_identification.ipynb)

############################################################################################################################

import cv2
import numpy as np
import os
from collections import deque

def compute_sift_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def compute_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

def most_common_element(buffer):
    count_dict = {}
    for item in buffer:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1
    most_common_item = max(count_dict, key=count_dict.get)
    return most_common_item

folder_path = 'signs'  
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]


descriptors_list = []
hu_moments_list = []
for image_path in image_paths:
    img = cv2.imread(image_path)
    _, descriptors = compute_sift_descriptors(img)
    hu_moments = compute_hu_moments(img)
    descriptors_list.append(descriptors)
    hu_moments_list.append(hu_moments)


cap = cv2.VideoCapture(0)
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()


best_match_buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)

    if descriptors_frame is not None:
        
        _, thresh_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY)
        moments_frame = cv2.moments(thresh_frame)
        hu_moments_frame = cv2.HuMoments(moments_frame).flatten()

        best_match_index = -1
        best_match_score = float('inf')
        #best_match_good_matches = None

        for i, descriptors in enumerate(descriptors_list):
            matches = bf.knnMatch(descriptors, descriptors_frame, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:  
                    good_matches.append(m)

        
            hu_distance = np.linalg.norm(hu_moments_list[i] - hu_moments_frame)

           
            if len(good_matches) > 25 and hu_distance < best_match_score:  
                best_match_index = i
                best_match_score = hu_distance
                #best_match_good_matches = good_matches

        if best_match_index != -1:
            best_match_buffer.append(best_match_index)

            if len(best_match_buffer) == best_match_buffer.maxlen:
                most_common_match = most_common_element(best_match_buffer)
                print(f"Most common match: {image_paths[most_common_match]}")

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
