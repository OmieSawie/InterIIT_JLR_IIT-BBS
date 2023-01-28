import cv2
import numpy as np

img_read = cv2.imread("./bookImg.jpg")

cap = cv2.VideoCapture("/dev/video2")

#features
sift = cv2.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img_read, None)

#make cv2 windows
cv2.namedWindow("Homography")
cv2.moveWindow("Homography", 10, 10)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

img_gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
img = img_read
cv2.drawKeypoints(img_gray, kp_image, img)

while True:
    _, frame = cap.read()
    if (frame is not None):

        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good_points.append(m)

        img3 = cv2.drawMatches(img_gray, kp_image, grayframe, kp_grayframe,
                               good_points, grayframe, 1)

        #homography
        if len(good_points) > 7:
            query_pts = np.float32([
                kp_image[m.queryIdx].pt for m in good_points
            ]).reshape(-1, 1, 2)
            train_pts = np.float32([
                kp_grayframe[m.trainIdx].pt for m in good_points
            ]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC,
                                              5.0)

            if matrix is not None:
                matches_mask = mask.ravel().tolist()

                #perspective transform
                h, w, _ = img.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                                  [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)

                homography = cv2.polylines(frame, [np.int32(dst)], True,
                                           (255, 0, 0), 3)

                cv2.imshow("Homography", homography)
        # else:
        # cv2.imshow("grayFrame", grayframe)

        cv2.drawKeypoints(grayframe, kp_grayframe, frame)
        # cv2.imshow("grayframe", grayframe)
        cv2.imshow("Image", img3)
        # cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
