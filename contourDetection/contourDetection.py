#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import math


def findContours(mask, blurred_frame):

    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        C = max(contours, key=cv2.contourArea)
        M = cv2.moments(C)

        if M["m00"] != 0:
            Gx = int(M["m10"] / M["m00"])
            Gy = int(M["m01"] / M["m00"])
        else:
            Gx = None
            Gy = None

        if Gx >= 350:
            print("right")

        if Gx < 350 and Gx > 290:
            print("straight")

        if Gx <= 290:
            print("left")

        cv2.circle(blurred_frame, (Gx, Gy), 5, (255, 255, 255), -1)

        cv2.putText(blurred_frame, "Centroid", (Gx - 25, Gy - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.drawContours(blurred_frame, C, -1, (0, 0, 200), 3)
    cv2.imshow('webcam feed', blurred_frame)
    cv2.imshow('mask', mask)


cap = cv2.VideoCapture('/dev/video0')
while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (640, 480))

    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])

    mask = cv2.inRange(frame, lower_black, upper_black)
    findContours(mask, blurred_frame)

    if cv2.waitKey(10) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
