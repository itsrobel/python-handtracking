#!/usr/bin/env python3


import cv2
cap = cv2.VideoCapture(-1)
import numpy as np
import time
import autopy

import handtracker as ht


width_cam, height_cam = 640, 480
width_screen , height_screen = autopy.screen.size()

cap.set(3,width_cam)
cap.set(4,height_cam)


previous_time,current_time  = 0, 0 

detector = ht.hand_detector(max_hands=1,detection_con=0.70)

while True:
    success, image = cap.read()
    image = cv2.flip(image,1)

    image = detector.find_hands(image)
    landmark_list,bounding_box = detector.find_position(image)
        


    if len(landmark_list) != 0:

        index_finger_x , index_finger_y = landmark_list[8][1:]
        mid_finger_x , mid_finger_y     = landmark_list[12][1:]

        
        fingers = detector.fingers_up()


        if fingers[1]==1 and fingers[2]==0:
            x3 = np.interp(index_finger_x , (0 , width_cam), (0, width_screen))
        
            y3 = np.interp(index_finger_y , (0 , height_cam), (0, height_screen))


            autopy.mouse.move(x3, y3)

    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time
    cv2.putText(image, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 4)
    cv2.imshow("image", image)
    cv2.waitKey(1)

