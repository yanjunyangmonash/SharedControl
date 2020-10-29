import numpy as np
import cv2

for frames in range(1574, 1575, 1):
    # Read input image
    frame = cv2.imread('Clip16_1M/clip16' + '_' + str(frames) + 'M.jpg')
    mask_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, contours_frame = cv2.threshold(mask_frame, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(contours_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for i in range(len(contours)):
        M = cv2.moments(contours[i], 0)
        if M['m00']:
            mass_centres_x = int(M['m10'] / M['m00'])
            mass_centres_y = int(M['m01'] / M['m00'])
        else:
            continue

        if mass_centres_x >= 506:
            mx = mass_centres_x
            my = mass_centres_y

            cv2.drawContours(frame, contours, i, (0, 0, 225), thickness=3)
            for j in range(len(contours[i])):
                dist = (contours[i][j][0][0] - 506) ** 2 + (
                        contours[i][j][0][1] - 279) ** 2

                if 401 ** 2 >= dist:
                    if 20 < contours[i][j][0][1] < 520:
                #if contours[i][j][0][0] < 684:
                        points.append(contours[i][j])

    # Setup K-means
    points = np.float32(points)
    points = np.array(points)
    clusterCount = 8  # Number of clusters to split the set by
    attempts = 10  #Number of times the algorithm is executed using different initial labels
    flags = cv2.KMEANS_PP_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(points, clusterCount, None, criteria, attempts, flags)
    a = [centers[0][0], centers[0][1]]
    print(centers.shape)
    new_centers = centers[centers != a]
    print(new_centers.shape)
    new_centers = new_centers.reshape([-1, 2])
    list_x = sorted(centers, key=lambda s: s[0], reverse=True)
    cw = list_x[0]
    ccw = list_x[1]
    cwl = []
    ccwl = []

    if list_x[1][1] > list_x[0][1]:
        cw = list_x[1]
        ccw = list_x[0]

    for p in list_x[2:]:
        if ((cw[1]-p[1]) ** 2 + (cw[0]-p[0]) ** 2) ** 0.5 < ((ccw[1]-p[1]) ** 2 + (ccw[0]-p[0]) ** 2) ** 0.5:
            cwl.append(p)
        else:
            ccwl.append(p)

    cv2.waitKey(0)
    for k in range(len(centers)):
        cv2.circle(frame, (int(centers[k][0]), int(centers[k][1])), radius=3, color=(0, 255, 0), thickness=-1)
    cv2.imshow('A', frame)
    cv2.waitKey(0)
    #cv2.imwrite('Clip16SL/clip16' + '_' + str(frames) + '.jpg', frame)

