#!/usr/bin/env python3

import cv2
cap = cv2.VideoCapture(-1)


import numpy as np
import handtracker as ht
import os
import time
import autopy

width_cam, height_cam = 680, 420
draw_color = (0 , 0 , 255)
brush_thickness = 15
cap.set(3,width_cam)
cap.set(4,height_cam)

previous_time,current_time  = 0, 0 

detector = ht.hand_detector(detection_con=0.70)


index_finger_x_previous , index_finger_y_previous = 0, 0
image_canvas = np.zeros((height_cam , width_cam,3 ), np.uint8)

while True:
    success, image = cap.read()

    image = cv2.flip(image,1)
    image = detector.find_hands(image)
    landmark_list,bounding_box = detector.find_position(image)

    if len(landmark_list) != 0:
        #print(landmark_list)


        index_finger_x , index_finger_y = landmark_list[8][1:]
        mid_finger_x , mid_finger_y     = landmark_list[12][1:]


        fingers = detector.fingers_up()
        #print(fingers)

        if fingers[1]:
            cv2.circle(image , (index_finger_x , index_finger_y), 15 , (255 , 0 ,0) , cv2.FILLED)
            if index_finger_x_previous == 0 and index_finger_y_previous ==0:
                index_finger_x_previous , index_finger_y_previous = index_finger_x, index_finger_y
                
            #cv2.line(image , (index_finger_x_previous , index_finger_y_previous), (index_finger_x , index_finger_y), draw_color , brush_thickness)
            cv2.line(image_canvas , (index_finger_x_previous , index_finger_y_previous), (index_finger_x , index_finger_y), draw_color , brush_thickness)

            index_finger_x_previous , index_finger_y_previous = index_finger_x, index_finger_y


        
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time
    cv2.putText(image, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 4)

    image_canvas_resized = cv2.resize(image_canvas, (image.shape[1], image.shape[0]))
    image = cv2.addWeighted(image, 0.5, image_canvas_resized, 0.5, 0)
    cv2.imshow("image", image)
    #cv2.imshow("canvas", image_canvas)
    cv2.waitKey(1)
