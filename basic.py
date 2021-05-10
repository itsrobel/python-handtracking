import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils


pre_time = 0
current_time = 0
while True:
    s, image = cap.read()

    image_rgb = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
       for hand_landmarks in results.multi_hand_landmarks:
           for id, landmarks in enumerate(hand_landmarks.landmark):
              #print(id, landmarks) 
              height, width , channels =image.shape

              center_x , center_y = int(landmarks.x*width), int(landmarks.y*height)
              print(id, center_x , center_y)

              # if id == 0:
              #     cv2.circle(image, (center_x, center_y), 25, (255 , 0, 255), cv2.FILLED)
           mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    current_time = time.time()

    frame_per_sec = 1/(current_time - pre_time)

    pre_time = current_time

    cv2.putText(image, str(int(frame_per_sec)) , (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0 , 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(1)
