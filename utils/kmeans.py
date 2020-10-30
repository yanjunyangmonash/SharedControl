import numpy as np
import cv2

def prep_for_Kmeans(contours_number, max_id, circle_x_coor, circle_y_coor, left_bound, right_bound, i_r, o_r, height):
    curve_pathx = []
    curve_pathy = []
    upbox_x = []
    upbox_y = []
    downbox_x = []
    downbox_y = []
    points_group = []

    for j in range(len(contours_number[max_id])):
        dist = (contours_number[max_id][j][0][0] - circle_x_coor) ** 2 + (
                contours_number[max_id][j][0][1] - circle_y_coor) ** 2
        if dist <= i_r ** 2:
            if height + 10 < contours_number[max_id][j][0][1] < 540 - (height + 10):
                points_group.append(contours_number[max_id][j])
        if i_r ** 2 <= dist <= o_r ** 2 and contours_number[max_id][j][0][0] > circle_x_coor:
            curve_pathx.append(contours_number[max_id][j][0][0])
            curve_pathy.append(contours_number[max_id][j][0][1])
        if (left_bound + 15 < contours_number[max_id][j][0][0] < right_bound - 15) and (
                0 <= contours_number[max_id][j][0][1] <= height):
            upbox_x.append(contours_number[max_id][j][0][0])
            upbox_y.append(contours_number[max_id][j][0][1])
        if (left_bound + 15 < contours_number[max_id][j][0][0] < right_bound - 15) and (
                540 - height <= contours_number[max_id][j][0][1] <= 540):
            downbox_x.append(contours_number[max_id][j][0][0])
            downbox_y.append(contours_number[max_id][j][0][1])

    return points_group, curve_pathx, curve_pathy, upbox_x, upbox_y, downbox_x, downbox_y


def kmeans_algorithm(points_group):
    points_group = np.float32(points_group)
    points_group = np.array(points_group)
    clusterCount = 12  # Number of clusters to split the set by
    attempts = 10  # Number of times the algorithm is executed using different initial labels
    flags = cv2.KMEANS_PP_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    if len(points_group) < clusterCount:
        return [], 0
    compactness, labels, centers = cv2.kmeans(points_group, clusterCount, None, criteria, attempts, flags)

    return centers, 1