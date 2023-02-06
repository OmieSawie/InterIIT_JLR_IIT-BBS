import cv2
import numpy as np
import math
import utils as ut
import time
from collections import deque

img_read = cv2.imread("./Socket No BG.png")  #Read the base image

cap = cv2.VideoCapture("/dev/video0")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# cap = cv2.VideoCapture("./Trial1Video")
assert cap.isOpened()

# cap = cv2.VideoCapture(0)

#features
sift = cv2.SIFT_create()  # initaite a SIFT Algorithm object
kp_image, desc_image = sift.detectAndCompute(
    img_read, None)  #Extract the features of the Base Image

#make cv2 windows
cv2.namedWindow("Homography")
cv2.moveWindow("Homography", 10, 20)
cv2.namedWindow("matchesframe")
cv2.moveWindow("matchesframe", 10, 400)
cv2.namedWindow("imgC")
cv2.moveWindow("imgC", 400, 400)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(
    index_params, search_params)  #Initiate a FlannBasedMatcher Object

img_gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
print(img_gray.shape[:2])
img = img_read
cv2.drawKeypoints(img_gray, kp_image, img)

Rot_deque = deque([], 20)
np_Final_Rot_deque = deque([], 20)
Trans_deque = deque([], 20)
np_Final_Trans_deque = deque([], 20)

while True:
    _, frame = cap.read()
    if (frame is not None):

        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgC = frame.copy()

        kp_grayframe, desc_grayframe = sift.detectAndCompute(
            grayframe, None)  #Apply SIFT feature extraction on the video frame

        if (desc_image is not None and len(desc_image) > 5
                and desc_grayframe is not None and len(desc_grayframe) > 5):
            matches = flann.knnMatch(
                desc_image, desc_grayframe, k=2
            )  #Apply the flann based KNN matcher between the base image and the video frame
        else:
            matches = []

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:  #Specify the threshold ratio between the distance of base image and video frame
                good_points.append(m)

        matchesframe = cv2.drawMatches(
            img_gray, kp_image, grayframe, kp_grayframe, good_points,
            grayframe, 1)  #Draw the matches for visual reference
        cv2.imshow("matchesframe", matchesframe)

        #homography
        if len(
                good_points
        ) > 7:  #Proceed ahead only if number of good matches exceeds a threshold
            query_pts = np.float32([
                kp_image[m.queryIdx].pt for m in good_points
            ]).reshape(-1, 1, 2)
            train_pts = np.float32([
                kp_grayframe[m.trainIdx].pt for m in good_points
            ]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC,
                                              5.0)

            if matrix is not None:

                ##########################################
                # A = np.matrix([[476.7, 0.0, 300.0], [0.0, 476.7, 300.0],
                #                [0.0, 0.0, 1.0]])
                A = np.matrix([
                    [245.31650564, 0.0, 313.80074017],
                    [0.0, 244.75854251, 235.72892421], [0.0, 0.0, 1.0]
                ])  #The camera matrix obtained from cameraCalibration
                (R, T) = ut.decHomography(
                    A, matrix
                )  #Derive the Rotation and Translation Data from "./utils.py"

                Rot = ut.decRotation(R)

                Rot_deque.append(Rot)
                Trans_deque.append(T)

                zR = np.matrix([[math.cos(Rot[2]), -math.sin(Rot[2])],
                                [math.sin(Rot[2]),
                                 math.cos(Rot[2])]])
                cv2.putText(
                    imgC, 'rX: {:0.2f} rY: {:0.2f} rZ: {:0.2f}'.format(
                        Rot[0] * 180 / np.pi, Rot[1] * 180 / np.pi,
                        Rot[2] * 180 / np.pi), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
                cv2.putText(
                    imgC, 'tX: {:0.2f} tY: {:0.2f} tZ: {:0.2f}'.format(
                        T[0, 0], T[0, 1], T[0, 2]), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
                pDot = np.dot((-200, -200), zR)
                red_point = (int(pDot[0, 0]), int(pDot[0, 1]))
                # cv2.circle(frame, (int(pDot[0, 0]) + train_pts[0],
                #                    int(pDot[0, 1]) + train_pts[1]), 5,
                #            (0, 0, 255), 2)
                # print(train_pts)
                #############################################
                # cv2.imshow("img", img)

                matches_mask = mask.ravel().tolist()

                #perspective transform
                h, w, _ = img.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                                  [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)

                homography = cv2.polylines(frame, [np.int32(dst)], True,
                                           (255, 0, 0), 3)

                # print(dst)

                cv2.imshow("Homography", homography)
        # else:
        # cv2.imshow("grayframe", grayframe)

        cv2.drawKeypoints(grayframe, kp_grayframe, frame)
        # cv2.imshow("img", grayframe)
        cv2.imshow("imgC", imgC)
        # cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            np_Rot_deque = np.array(Rot_deque)
            np_Trans_deque = np.array(Trans_deque)
            mean_Rot = np.mean(
                np_Rot_deque, axis=0
            )  #Find the Mean and Standard Deviation of the data from 20 frames
            mean_Trans = np.mean(np_Trans_deque, axis=0)
            sd_Rot = np.std(np_Rot_deque, axis=0)
            sd_Trans = np.std(np_Trans_deque, axis=0)

            for x in range(20):  #Iterate through a buffer of 20 frames
                for y in range(3):
                    if (np_Rot_deque[x][y] < mean_Rot[y] - 1 * sd_Rot[y]
                            or np_Rot_deque[x][y] > mean_Rot[y] + 1 * sd_Rot[y]
                        ):  #Eliminate the Rotation data with high deviation
                        break
                    if (np_Trans_deque[x][0][y] <
                            mean_Trans[0][y] - 1 * sd_Trans[0][y]
                            or np_Trans_deque[x][0][y] >
                            mean_Trans[0][y] + 1 * sd_Trans[0][y]
                        ):  #Eliminate the Translation Data with high deviation
                        break

                    np_Final_Rot_deque.append(np_Rot_deque[x])
                    np_Final_Trans_deque.append(np_Trans_deque[x])

            np_Final_Rot_deque = np.multiply(np_Final_Rot_deque, 180 / np.pi)
            np_Final_Rot_deque = np.round(np_Final_Rot_deque, 3)
            np_Final_Trans_deque = np.round(np_Final_Trans_deque, 3)

            np_Output_Rot_Data = np.mean(
                np.array(np_Final_Rot_deque),
                axis=0)  #calculate the mean of the filtered data
            np_Output_Trans_Data = np.mean(np.array(np_Final_Trans_deque),
                                           axis=0)
            print("np_Output_Rot_Data", np_Output_Rot_Data)
            print("np_Output_Trans_Data", np_Output_Trans_Data)
            break

cap.release()
cv2.destroyAllWindows()
