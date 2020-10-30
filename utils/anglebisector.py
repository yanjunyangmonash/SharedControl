from math import atan, tan, pi

def find_angle_bisector(line1_p1, line1_p2, line2_p1, line2_p2, centers_matrix):
    # line1 points need to give points on the right
    if (line1_p1[0] - line1_p2[0]) == 0:
        return 0, (0, 0)
    k1 = (line1_p1[1] - line1_p2[1]) / (line1_p1[0] - line1_p2[0])
    angle1 = atan(k1)
    b1 = b2 = -1
    c1 = line1_p1[1] - k1 * line1_p1[0]
    # Use perpendicular point to prevent noise (frame627, which choose right point as the next point)
    pvx = (b1 * b1 * line2_p1[0] - k1 * b1 * line2_p1[1] - k1 * c1) / (k1 * k1 + b1 * b1)
    pvy = (-k1 * b1 * line2_p1[0] + k1 * k1 * line2_p1[1] - b1 * c1) / (k1 * k1 + b1 * b1)

    min_line2_p2_dist = 1000
    # If the nearest point pointing to a wrong direction (on the right of the line line2_p1 to pv)
    if (line2_p1[0] - line2_p2[0]) * (pvy - line2_p2[1]) - (line2_p1[1] - line2_p2[1]) * (pvx - line2_p2[0]) > 0:
        wrong_point = line2_p2[0]
        for k in range(len(centers_matrix)):
            new_line2_p2_dist = (([k][0] - line2_p1[0]) ** 2 + (
                    centers_matrix[k][1] - line2_p1[1]) ** 2) ** 0.5
            if new_line2_p2_dist < min_line2_p2_dist and new_line2_p2_dist != 0 and centers_matrix[k][0] != wrong_point:
                new_line2_p2x = centers_matrix[k][0]
                new_line2_p2y = centers_matrix[k][1]
                min_line2_p2_dist = new_line2_p2_dist
        line2_p2 = (new_line2_p2x, new_line2_p2y)

    if (line2_p2[0] - line2_p1[0]) == 0:
        k2 = 0
    else:
        k2 = (line2_p2[1] - line2_p1[1]) / (line2_p2[0] - line2_p1[0])
    angle2 = atan(k2)

    if angle1 < 0:
        angle1 = angle1+pi
    if angle2 < 0:
        angle2 = angle2 + pi
    intersrct_angle = angle2-angle1
    bisector_line_angle = angle1+intersrct_angle/2
    k5 = tan(bisector_line_angle)

    a1 = k1
    a2 = k2
    c2 = line2_p1[1] - k2 * line2_p1[0]

    '''
    if a2 > a1:
        a5 = (a2 - a1) / 2 + a1
    else:
        a5 = (a1 - a2) / 2 + a2
    a1 = k1
    a2 = k2
    c2 = line2_p1[1] - k2 * line2_p1[0]
    k5 = tan(a5)
    '''


    if (a1 * b2 - a2 * b1) == 0 or (b1 * a2 - b2 * a1) == 0:
        return k5, (0, 0)

    p5 = ((b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1), (a1 * c2 - a2 * c1) / (b1 * a2 - b2 * a1))

    return k5, p5


def find_angle_bisector_curve(line1_p1, line1_p2, line2_p1, line2_p2, centers_matrix):
    # line1 points need to give points on the right
    if (line1_p1[0] - line1_p2[0]) == 0:
        return 0, (0, 0)
    k1 = (line1_p1[1] - line1_p2[1]) / (line1_p1[0] - line1_p2[0])
    a1 = atan(k1)
    b1 = b2 = -1
    c1 = line1_p1[1] - k1 * line1_p1[0]
    # Use perpendicular point to prevent noise (frame627, which choose right point as the next point)
    pvx = (b1 * b1 * line2_p1[0] - k1 * b1 * line2_p1[1] - k1 * c1) / (k1 * k1 + b1 * b1)
    pvy = (-k1 * b1 * line2_p1[0] + k1 * k1 * line2_p1[1] - b1 * c1) / (k1 * k1 + b1 * b1)

    min_line2_p2_dist = 1000
    # If the nearest point pointing to a wrong direction (on the right of the line line2_p1 to pv)
    if (line2_p1[0] - line2_p2[0]) * (pvy - line2_p2[1]) - (line2_p1[1] - line2_p2[1]) * (pvx - line2_p2[0]) > 0:
        wrong_point = line2_p2[0]
        for k in range(len(centers_matrix)):
            new_line2_p2_dist = (([k][0] - line2_p1[0]) ** 2 + (
                    centers_matrix[k][1] - line2_p1[1]) ** 2) ** 0.5
            if new_line2_p2_dist < min_line2_p2_dist and new_line2_p2_dist != 0 and centers_matrix[k][0] != wrong_point:
                new_line2_p2x = centers_matrix[k][0]
                new_line2_p2y = centers_matrix[k][1]
                min_line2_p2_dist = new_line2_p2_dist
        line2_p2 = (new_line2_p2x, new_line2_p2y)

    if (line2_p2[0] - line2_p1[0]) == 0:
        k2 = 0
    else:
        k2 = (line2_p2[1] - line2_p1[1]) / (line2_p2[0] - line2_p1[0])
    a2 = atan(k2)

    if a2 > a1:
        a5 = (a2 - a1) / 2 + a1
    else:
        a5 = (a1 - a2) / 2 + a2
    a1 = k1
    a2 = k2
    c2 = line2_p1[1] - k2 * line2_p1[0]
    k5 = tan(a5)

    if (a1 * b2 - a2 * b1) == 0 or (b1 * a2 - b2 * a1) == 0:
        return k5, (0, 0)

    p5 = ((b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1), (a1 * c2 - a2 * c1) / (b1 * a2 - b2 * a1))

    return k5, p5


def find_angle_bisector_curveedge(line1_p1, line1_p2, line2_p1, line2_p2, centers_matrix):
    # line1 points need to give points on the right
    if (line1_p1[0] - line1_p2[0]) == 0:
        return 0, (0, 0)
    k1 = (line1_p1[1] - line1_p2[1]) / (line1_p1[0] - line1_p2[0])
    angle1 = atan(k1)
    b1 = b2 = -1
    c1 = line1_p1[1] - k1 * line1_p1[0]
    # Use perpendicular point to prevent noise (frame627, which choose right point as the next point)
    pvx = (b1 * b1 * line2_p1[0] - k1 * b1 * line2_p1[1] - k1 * c1) / (k1 * k1 + b1 * b1)
    pvy = (-k1 * b1 * line2_p1[0] + k1 * k1 * line2_p1[1] - b1 * c1) / (k1 * k1 + b1 * b1)

    min_line2_p2_dist = 1000
    # If the nearest point pointing to a wrong direction (on the right of the line line2_p1 to pv)
    if (line2_p1[0] - line2_p2[0]) * (pvy - line2_p2[1]) - (line2_p1[1] - line2_p2[1]) * (pvx - line2_p2[0]) > 0:
        wrong_point = line2_p2[0]
        for k in range(len(centers_matrix)):
            new_line2_p2_dist = (([k][0] - line2_p1[0]) ** 2 + (
                    centers_matrix[k][1] - line2_p1[1]) ** 2) ** 0.5
            if new_line2_p2_dist < min_line2_p2_dist and new_line2_p2_dist != 0 and centers_matrix[k][0] != wrong_point:
                new_line2_p2x = centers_matrix[k][0]
                new_line2_p2y = centers_matrix[k][1]
                min_line2_p2_dist = new_line2_p2_dist
        line2_p2 = (new_line2_p2x, new_line2_p2y)

    if (line2_p2[0] - line2_p1[0]) == 0:
        k2 = 0
    else:
        k2 = (line2_p2[1] - line2_p1[1]) / (line2_p2[0] - line2_p1[0])
    angle2 = atan(k2)

    if angle1 < 0:
        angle1 = angle1 + pi
    if angle2 < 0:
        angle2 = angle2 + pi

    if k1*k2 > 0:
        intersrct_angle = angle2 - angle1
        bisector_line_angle = angle1 + intersrct_angle / 2
    else:
        angle1_inner = pi - angle1
        intersrct_angle = angle2 + angle1_inner
        bisector_line_angle = angle1 + intersrct_angle / 2

    k5 = tan(bisector_line_angle)
    a1 = k1
    a2 = k2
    c2 = line2_p1[1] - k2 * line2_p1[0]

    if (a1 * b2 - a2 * b1) == 0 or (b1 * a2 - b2 * a1) == 0:
        return k5, (0, 0)

    p5 = ((b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1), (a1 * c2 - a2 * c1) / (b1 * a2 - b2 * a1))

    return k5, p5