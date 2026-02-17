import cv2 as cv
import numpy as np
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
mphands = mp.solutions.hands
mpdraw = mp.solutions.drawing_utils
hands = mphands.Hands()
ptime = 0

while True:
    success,img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img,handlms,mphands.HAND_CONNECTIONS)
            fingers = []
            for id,lm in enumerate(handlms.landmark):
                 fingers.append((id,lm.x,lm.y))
                 h, w, c = img.shape
                 cx, cy = int(lm.x * w), int(lm.y * h)
                 if id == 8:  # index fingertip
                    cv.circle(img, (cx, cy), 10, (255, 0, 0), 2)
            if fingers[8][2] < fingers[6][2] :
                print("index finger is up")
            else:
                print("Index finger is down")       
                # print(id,lm)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    img1 = cv.flip(img,1)
    cv.imshow("Image",img1)
    cv.waitKey(1)

