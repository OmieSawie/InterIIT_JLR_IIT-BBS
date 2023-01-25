import cv2
import numpy as np
import math 


def findContours(mask,blurred_frame):

    contours,_ = cv2.findContours(mask,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
     

    for contour in contours:
        # C = max(contours,key=cv2.contourArea)

        M = cv2.moments(contour)

        if M["m00"]!=0 and cv2.contourArea(contour)>6000:
            Gx = int(M["m10"] / M["m00"])
            Gy = int(M["m01"] / M["m00"])
            
            cv2.drawContours(blurred_frame,contour,-1,(0,0,100),2)
            
            cv2.circle(blurred_frame, (Gx, Gy), 7, (255, 0, 0), -1)
 
            cv2.putText(blurred_frame, "Centroid", (Gx - 25, Gy - 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 00, 100), 2)


        else:
            Gx=None
            Gy=None
            
        # if Gx >= 350:
            # print("right")
# 
        # if Gx < 350 and Gx > 290:
            # print("straight")
                # 
        # if Gx <= 290:
            # print("left")

    cv2.imshow('webcam feed' , blurred_frame)
    cv2.imshow('mask',mask)
    cv2.imwrite("outpput_by_contour.png",blurred_frame)


# cap = cv2.VideoCapture(0)
while True: 
    # ret, frame = cap.read()
    
    frame = cv2.imread("theme_arena_initial.png")
    # frame = cv2.resize(frame,(640,480))

    blurred_frame = cv2.GaussianBlur(frame,(5,5),0)
    
    lower_black = np.array([0,100,0])
    upper_black = np.array([0,255,255])

    mask = cv2.inRange(frame,lower_black,upper_black)
    findContours(mask,blurred_frame)
 

    if cv2.waitKey(10) & 0xFF == ord(' '):
        break
    
# cap.release()
cv2.destroyAllWindows()

