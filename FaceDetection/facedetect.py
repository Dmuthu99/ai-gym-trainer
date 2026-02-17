import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
cTime = 0
pTime = 0

mpfaceDetection = mp.solutions.face_detection
mpdraw = mp.solutions.drawing_utils
faceDetection = mpfaceDetection.FaceDetection(0.75)

while True:
    succes,img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print((results.detections))
    if results.detections:
        for id,detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box
            xmin,ymin,width,height = bbox.xmin,bbox.ymin,bbox.width,bbox.height
            hi,wi,ci = img.shape
            bbox_updated = (
                int(wi*xmin),
                int(hi*ymin),
                int(wi*width),
                int(hi*height)
            )
            x,y,w,h = bbox_updated
            d1 = x + w
            d2 = y + h
            cv.rectangle(img,(x,y),(d1,d2),(255,0,255),2)
            cv.putText(img,f'score:{detection.score[0]:.2f}',(x,y-10),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),3)

    img1 = cv.flip(img,1)
    cv.imshow("Image",img)
    cv.waitKey(1)

    # cTime = time.time()
    # fps = 1/(cTime-pTime)
    # pTime = cTime
    # cv.putText(img,f'fps{int(fps)}',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)