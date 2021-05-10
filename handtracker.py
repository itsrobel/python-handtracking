#!/usr/bin/env python3


import cv2
import mediapipe as mp
import math
import time



class hand_detector():
    def __init__(self, mode=False, max_hands= 2 , detection_con = 0.5, track_con = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con



        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode , self.max_hands, self.detection_con, self.track_con)

        self.mp_draw = mp.solutions.drawing_utils

        self.tip_ids = [4,8,12,16,20]

    def find_hands(self, image, draw=True):

        image_rgb = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(image_rgb)

        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:

                if draw:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return image
    def find_position(self, image, hand_index=0 , draw = True):

        self.bounding_box  = []
        self.landmark_list = []
    


        if self.results.multi_hand_landmarks:
            chosen_hand = self.results.multi_hand_landmarks[hand_index]
            for id, landmarks in enumerate(chosen_hand.landmark):
                #print(id, landmarks) 
                height, width , channels =image.shape

                center_x , center_y = int(landmarks.x*width), int(landmarks.y*height)
                #print(id, center_x , center_y)
                self.landmark_list.append([id, center_x, center_y])

                # if id == 0:
                # cv2.circle(image, (center_x, center_y), 25, (255 , 0, 255), cv2.FILLED)
                # if draw:
                #     self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return self.landmark_list, self.bounding_box

    def fingers_up(self):
        fingers = []

        if self.landmark_list[self.tip_ids[0]][1] < self.landmark_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)

        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.landmark_list[self.tip_ids[id]][2] < self.landmark_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self , p1, p2,  image, draw=True,r=15 , t=3):
        x1 , y1 = self.landmark_list[p1][1:]
        x2 , y2 = self.landmark_list[p2][1:]

        cx , cy = (x1+x2)// 2, (y1+y2) //2

        if draw:
            cv2.line(image, (x1, y1), (x2, y2), (255,0,255), t)
            cv2.circle(image, (x1,y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(image, (x2,y2), r, (255,0,255), cv2.FILLED)
            cv2.circle(image, (cx,cy), r, (255,0,255), cv2.FILLED)
                
        length = math.hypot(x2-x1, y2-y1)

        return length, image, [x1,y1, x2,y2,cx,cy]


cap = cv2.VideoCapture(0)
def main():

    pre_time = 0
    current_time = 0

    detector = hand_detector()
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

if __name__ == "__main__":
    main()
