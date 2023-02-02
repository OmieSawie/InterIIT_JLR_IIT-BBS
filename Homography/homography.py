import cv2
import numpy as np
import math
import utils as ut
import time

img_read = cv2.imread("./Socket No BG.png")

cap = cv2.VideoCapture("/dev/video2")
assert cap.isOpened()

# cap = cv2.VideoCapture(0)

#features
sift = cv2.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img_read, None)

#make cv2 windows
cv2.namedWindow("Homography")
cv2.moveWindow("Homography", 10, 20)
cv2.namedWindow("matchesframe")
cv2.moveWindow("matchesframe", 10, 400)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

img_gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
img = img_read
cv2.drawKeypoints(img_gray, kp_image, img)

centroid = []

while True:
    _, frame = cap.read()
    if (frame is not None):

        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgC = frame.copy()

        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)

        if (desc_image is not None and len(desc_image) > 2
                and desc_grayframe is not None and len(desc_grayframe) > 2):
            matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
        else:
            matches = []

        good_points = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good_points.append(m)

        matchesframe = cv2.drawMatches(img_gray, kp_image, grayframe,
                                       kp_grayframe, good_points, grayframe, 1)
        cv2.imshow("matchesframe", matchesframe)

        #homography
        if len(good_points) > 10:
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
                A = np.matrix([[476.7, 0.0, 400.0], [0.0, 476.7, 400.0],
                               [0.0, 0.0, 1.0]])
                (R, T) = ut.decHomography(A, matrix)

                Rot = ut.decRotation(R)
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

                centroid.append(np.int32(dst))

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
            # print(centroid)
            # np_centroid = np.array(centroid)
            # mean = np.mean(np_centroid, axis=0)
            # sd = np.std(np_centroid, axis=0)
            # final_list = [x for x in centroid if (x > mean - 2 * sd)]
            # final_list = [x for x in final_list if (x < mean + 2 * sd)]
            # print(final_list)
            break

cap.release()
cv2.destroyAllWindows()
