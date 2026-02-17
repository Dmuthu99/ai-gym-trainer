import cv2 as cv
import mediapipe as mp
import numpy as np
import poseEstimation.poseModule as pose

cap = cv.VideoCapture(0)
c1 = pose.poseDetect()
dir1 = 0
dir2 = 0
rcount = 0
lcount = 0
while True:
    success,img = cap.read()
    if not success or img is None:
        continue
    c1.poseDetection(img)
    h,w,c = img.shape
    pose = c1.findPose(img)
    right = c1.findAngle(img,16,14,12)
    left = c1.findAngle(img,11,13,15)
    min = 15
    max = 170
    right_range = np.interp(right,(min,max),(0,100))
    left_range = np.interp(left,(min,max),(0,100)) 

    if (right_range <= 10) :
        if (dir1 == 0): 
            rcount += 0.5
            dir1 = 1

    if (right_range > 90) :
        if (dir1 == 1):
            rcount += 0.5
            dir1 = 0

    if (left_range <= 10) :
        if (dir2 == 0): 
            lcount += 0.5
            dir2 = 1

    if (left_range > 90) :
        if (dir2 == 1):
            lcount += 0.5
            dir2 = 0         

    cv.putText(img,f'right: {int(rcount)}',(w-550,30),cv.FONT_HERSHEY_PLAIN,2,(0,255,0), 2)       
    cv.putText(img,f'left: {int(lcount)}',(w-150,30),cv.FONT_HERSHEY_PLAIN,2,(0,255,0), 2)

    cv.imshow("Image",img)  
    cv.waitKey(1)
