#!/usr/bin/env python3
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(-1)

import handtracker as ht

pre_time = 0
current_time = 0

detector = ht.hand_detector()
while True:
    s, image = cap.read()
    image = detector.find_hands(image)
    hand_position = detector.find_position(image)

    if len(hand_position) != 0:
        print(hand_position[2])
    current_time = time.time()

    frame_per_sec = 1/(current_time - pre_time)

    pre_time = current_time

    cv2.putText(image, str(int(frame_per_sec)) , (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0 , 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(1)

