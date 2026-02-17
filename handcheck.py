import cv2 as cv
import numpy as np
import mediapipe as mp
import time 
import handmodule as hm  


ptime = 0
cap = cv.VideoCapture(0)
detector = hm.handDetection()
    

while True:
    success,img = cap.read()
    img = detector.findhands(img)
    positions = detector.findposition(img)
    if len(positions) != 0:
        id,xi,yi = positions[8]
        id,xp,yp = positions[20]
        print(f"x-position {abs(xi-xp)}")
        print(f"y-position {abs(yi-yp)}")

        if(abs(xi-xp) > abs(yi-yp)):
            print("bicep curl")
        else:
            print("hammer curl")

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    img1 = cv.flip(img,1)
    cv.imshow("Image",img1)
    cv.waitKey(1)