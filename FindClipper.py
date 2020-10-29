import cv2
import openpyxl
from math import atan, tan, asin, pi
import numpy as np
import constant

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

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


def get_symmetric_point_a(tip_point, edge_point1, edge_point2, k_bisector):
    # Situation a is for bottom; top and curve boundary
    edge_point1_dist = ((edge_point1[1] - tip_point[1]) ** 2 + (edge_point1[0] - tip_point[0]) ** 2) ** 0.5
    edge_point2_dist = ((edge_point2[1] - tip_point[1]) ** 2 + (edge_point2[0] - tip_point[0]) ** 2) ** 0.5

    if edge_point2_dist > edge_point1_dist:
        edge_point1x = ((1 - k_bisector ** 2) * edge_point2[0] - 2 * k_bisector * (-1) * edge_point2[
            1] - 2 * k_bisector * (tip_point[1] - k_bisector * tip_point[0])) / (
                               1 + k_bisector ** 2)
        edge_point1y = ((k_bisector ** 2 - 1) * edge_point2[1] - 2 * k_bisector * (-1) * edge_point2[0] - 2 * (-1) * (
                tip_point[1] - k_bisector * tip_point[0])) / (
                               1 + k_bisector ** 2)
        edge_point1 = (edge_point1x, edge_point1y)
    else:
        edge_point2x = ((1 - k_bisector ** 2) * edge_point1[0] - 2 * k_bisector * (-1) * edge_point1[
            1] - 2 * k_bisector * (tip_point[1] - k_bisector * tip_point[0])) / (
                               1 + k_bisector ** 2)
        edge_point2y = ((k_bisector ** 2 - 1) * edge_point1[1] - 2 * k_bisector * (-1) * edge_point1[0] - 2 * (-1) * (
                tip_point[1] - k_bisector * tip_point[0])) / (
                               1 + k_bisector ** 2)
        edge_point2 = (edge_point2x, edge_point2y)

    return edge_point1, edge_point2


def get_symmetric_point_b(tip_point, edge_point1, k_bisector):
    # Situation b is for bottom&curve and top&curve boundary
    edge_point2x = ((1 - k_bisector ** 2) * edge_point1[0] - 2 * k_bisector * (-1) * edge_point1[1] - 2 * k_bisector * (
                tip_point[1] - k_bisector * tip_point[0])) / (
                           1 + k_bisector ** 2)
    edge_point2y = ((k_bisector ** 2 - 1) * edge_point1[1] - 2 * k_bisector * (-1) * edge_point1[0] - 2 * (-1) * (
                tip_point[1] - k_bisector * tip_point[0])) / (
                           1 + k_bisector ** 2)
    edge_point2 = (edge_point2x, edge_point2y)

    return edge_point2


def get_feature_points_a(centers_matrix, circle_rad, circle_x_coor, circle_y_coor):
    # Situation a is for curve boundary
    list_dist_to_center = []
    for m in range(len(centers_matrix)):
        dist_to_center = circle_rad - abs(
            ((centers_matrix[m][0] - circle_x_coor) ** 2 + (centers_matrix[m][1] - circle_y_coor) ** 2) ** 0.5)
        list_dist_to_center.append(dist_to_center)
    sorted_list = sorted(list_dist_to_center)
    max_number = list_dist_to_center.index(sorted_list[0])
    sec_max_number = list_dist_to_center.index(sorted_list[1])
    next_max_point = get_next_point(centers_matrix, (centers_matrix[max_number][0], centers_matrix[max_number][1]))
    if next_max_point[1] == centers_matrix[sec_max_number][1]:
        sec_max_number = list_dist_to_center.index(sorted_list[2])
    next_sec_max_point = get_next_point(centers_matrix,
                                        (centers_matrix[sec_max_number][0], centers_matrix[sec_max_number][1]),
                                        restriction_point=(
                                        centers_matrix[max_number][0], centers_matrix[max_number][1]))

    if centers_matrix[max_number][1] < centers_matrix[sec_max_number][1]:
        right_top_point = (centers_matrix[max_number][0], centers_matrix[max_number][1])
        next_right_top_point = next_max_point
        right_bottom_point = (centers_matrix[sec_max_number][0], centers_matrix[sec_max_number][1])
        next_right_bottom_point = next_sec_max_point
    else:
        right_top_point = (centers_matrix[sec_max_number][0], centers_matrix[sec_max_number][1])
        next_right_top_point = next_sec_max_point
        right_bottom_point = (centers_matrix[max_number][0], centers_matrix[max_number][1])
        next_right_bottom_point = next_max_point

    return right_top_point, next_right_top_point, right_bottom_point, next_right_bottom_point


def get_feature_points_b(centers_matrix, bottom_or_top=1):
    # Situation b is for bottom&curve and top&curve boundary
    list_points_x = sorted(centers_matrix, key=lambda s: s[0])
    list_points_y = sorted(centers_matrix, key=lambda s: s[1])
    right_point = (list_points_x[-1][0], list_points_x[-1][1])
    if bottom_or_top:
        # Find bottom point
        top_or_bottom_point = (list_points_y[-1][0], list_points_y[-1][1])
        if top_or_bottom_point[1] <= right_point[1]:
            top_or_bottom_point = (list_points_y[-2][0], list_points_y[-2][1])
    else:
        # Find top point
        top_or_bottom_point = (list_points_y[0][0], list_points_y[0][1])
        if top_or_bottom_point[1] >= right_point[1]:
            top_or_bottom_point = (list_points_y[1][0], list_points_y[1][1])

    return right_point, top_or_bottom_point


def get_feature_points_c(centers_matrix, right_boundary, left_boundary, bottom_or_top=1):
    # Situation c is for bottom and top boundary
    if bottom_or_top:
        list_points_y = sorted(centers_matrix, key=lambda s: s[1], reverse=True)
        bottom_points = [list_points_y[0], list_points_y[1], list_points_y[2]]
        list_bottom_points_x = sorted(bottom_points, key=lambda s: s[0], reverse=True)

        mid_to_right = list_bottom_points_x[0][0] - list_bottom_points_x[1][0]
        mid_to_left = list_bottom_points_x[2][0] - list_bottom_points_x[1][0]

        if mid_to_right < mid_to_left:
            if list_bottom_points_x[0][1] < list_bottom_points_x[1][1]:
                right_bottom_point = (list_bottom_points_x[1][0], list_bottom_points_x[1][1])
                del list_bottom_points_x[1]
            else:
                right_bottom_point = (list_bottom_points_x[0][0], list_bottom_points_x[0][1])
                del list_bottom_points_x[0]
        else:
            right_bottom_point = (list_bottom_points_x[0][0], list_bottom_points_x[0][1])
            del list_bottom_points_x[0]
        next_right_bottom_point = get_next_point(centers_matrix, right_bottom_point, add_x_restriction=1)

        dist1 = (next_right_bottom_point[0] - list_bottom_points_x[0][0]) ** 2 + (
                next_right_bottom_point[1] - list_bottom_points_x[0][1]) ** 2
        dist2 = (next_right_bottom_point[0] - list_bottom_points_x[1][0]) ** 2 + (
                next_right_bottom_point[1] - list_bottom_points_x[1][1]) ** 2
        if dist1 < dist2:
            del list_bottom_points_x[0]
        else:
            del list_bottom_points_x[1]
        bottom_point = (list_bottom_points_x[0][0], list_bottom_points_x[0][1])
        next_bottom_point = get_next_point(centers_matrix, bottom_point, add_x_restriction=0)

        '''
        right_bottom_point = (list_bottom_points_x[0][0], list_bottom_points_x[0][1])
        next_right_bottom_point = get_next_point(centers_matrix, right_bottom_point, add_x_restriction=1)
        if bottom_points[1][1] != next_right_bottom_point[1]:
            bottom_point = (bottom_points[1][0], bottom_points[1][1])
        else:
            bottom_point = (bottom_points[2][0], bottom_points[2][1])
        next_bottom_point = get_next_point(centers_matrix, bottom_point, add_x_restriction=0)
        '''

        output_point1 = right_bottom_point
        output_point2 = next_right_bottom_point
        output_point3 = bottom_point
        output_point4 = next_bottom_point

    else:
        list_points_y = sorted(centers_matrix, key=lambda s: s[1])
        top_points = [list_points_y[0], list_points_y[1], list_points_y[2]]
        list_top_points_x = sorted(top_points, key=lambda s: s[0], reverse=True)

        mid_to_right = list_top_points_x[0][0] - list_top_points_x[1][0]
        mid_to_left = list_top_points_x[2][0] - list_top_points_x[1][0]

        if mid_to_right < mid_to_left:
            if list_top_points_x[0][1] < list_top_points_x[1][1]:
                right_top_point = (list_top_points_x[0][0], list_top_points_x[0][1])
                del list_top_points_x[0]
            else:
                right_top_point = (list_top_points_x[1][0], list_top_points_x[1][1])
                del list_top_points_x[1]
        else:
            right_top_point = (list_top_points_x[0][0], list_top_points_x[0][1])
            del list_top_points_x[0]
        next_right_top_point = get_next_point(centers_matrix, right_top_point, add_x_restriction=1)

        dist1 = (next_right_top_point[0] - list_top_points_x[0][0]) ** 2 + (
                    next_right_top_point[1] - list_top_points_x[0][1]) ** 2
        dist2 = (next_right_top_point[0] - list_top_points_x[1][0]) ** 2 + (
                    next_right_top_point[1] - list_top_points_x[1][1]) ** 2
        if dist1 < dist2:
            del list_top_points_x[0]
        else:
            del list_top_points_x[1]
        top_point = (list_top_points_x[0][0], list_top_points_x[0][1])
        next_top_point = get_next_point(centers_matrix, top_point, add_x_restriction=0)
        '''
        right_top_point = (list_top_points_x[0][0], list_top_points_x[0][1])
        next_right_top_point = get_next_point(centers_matrix, right_top_point, add_x_restriction=1)
        if top_points[1][1] != next_right_top_point[1]:
            top_point = (top_points[1][0], top_points[1][1])
        else:
            top_point = (top_points[2][0], top_points[2][1])
        next_top_point = get_next_point(centers_matrix, top_point, add_x_restriction=0)
        '''

        output_point1 = top_point
        output_point2 = next_top_point
        output_point3 = right_top_point
        output_point4 = next_right_top_point

    return output_point1, output_point2, output_point3, output_point4


def get_next_point(centers_matrix, end_point, add_x_restriction=1, restriction_point=(0, 0)):
    next_point = (0, 0)
    min_dist = 1000
    if add_x_restriction:
        if restriction_point == (0, 0):
            for k in range(len(centers_matrix)):
                dist_to_next_point = ((centers_matrix[k][0] - end_point[0]) ** 2 + (
                            centers_matrix[k][1] - end_point[1]) ** 2) ** 0.5
                if dist_to_next_point < min_dist and dist_to_next_point != 0 and abs(
                        (end_point[1] - centers_matrix[k][1])) < 130 and centers_matrix[k][0] < end_point[0]:
                    next_point = (centers_matrix[k][0], centers_matrix[k][1])
                    min_dist = dist_to_next_point
        else:
            for k in range(len(centers_matrix)):
                dist_to_next_point = ((centers_matrix[k][0] - end_point[0]) ** 2 + (
                            centers_matrix[k][1] - end_point[1]) ** 2) ** 0.5
                if dist_to_next_point < min_dist and dist_to_next_point != 0 and abs(
                        (end_point[1] - centers_matrix[k][1])) < 130 and centers_matrix[k][0] < end_point[0] and \
                        centers_matrix[k][1] != restriction_point[1]:
                    next_point = (centers_matrix[k][0], centers_matrix[k][1])
                    min_dist = dist_to_next_point
    else:
        if restriction_point == (0, 0):
            for k in range(len(centers_matrix)):
                dist_to_next_point = ((centers_matrix[k][0] - end_point[0]) ** 2 + (
                            centers_matrix[k][1] - end_point[1]) ** 2) ** 0.5
                if dist_to_next_point < min_dist and dist_to_next_point != 0 and abs(
                        (end_point[1] - centers_matrix[k][1])) < 130:
                    next_point = (centers_matrix[k][0], centers_matrix[k][1])
                    min_dist = dist_to_next_point
        else:
            for k in range(len(centers_matrix)):
                dist_to_next_point = ((centers_matrix[k][0] - end_point[0]) ** 2 + (
                            centers_matrix[k][1] - end_point[1]) ** 2) ** 0.5
                if dist_to_next_point < min_dist and dist_to_next_point != 0 and abs(
                        (end_point[1] - centers_matrix[k][1])) < 130 and centers_matrix[k][1] != restriction_point[1]:
                    next_point = (centers_matrix[k][0], centers_matrix[k][1])
                    min_dist = dist_to_next_point

    return next_point


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
        if (left_bound + 15 < contours[max_id][j][0][0] < right_bound - 15) and (
                540 - height <= contours[max_id][j][0][1] <= 540):
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


def get_tool_length(point2, point3, centers_matrix, k_val):
    if point2[0] > point3[0]:
        mid_point23 = (int(point3[0] + (point2[0] - point3[0]) / 2), int(point3[1] + (point2[1] - point3[1]) / 2))
    else:
        mid_point23 = (int(point2[0] + (point3[0] - point2[0]) / 2), int(point2[1] + (point3[1] - point2[1]) / 2))
    dist_list = []
    for point_id in range(len(centers_matrix)):
        dist = ((centers_matrix[point_id][0] - mid_point23[0]) ** 2 + (
                    centers_matrix[point_id][1] - mid_point23[1]) ** 2) ** 0.5
        dist_list.append(dist)
    tip_id = dist_list.index(max(dist_list))
    tip_point = (centers_matrix[tip_id][0], centers_matrix[tip_id][1])

    a1 = k_val
    b1 = -1
    c1 = tip_point[1] - k_val * tip_point[0]
    if point2[0] - point3[0] == 0:
        return tip_point, (0, 0), 0
    a2 = (point2[1] - point3[1]) / (point2[0] - point3[0])
    b2 = -1
    c2 = point2[1] - a2 * point2[0]

    if (a1 * b2 - a2 * b1) == 0 or (b1 * a2 - b2 * a1) == 0:
        return tip_point, (0, 0), 0

    intersect_point = ((b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1), (a1 * c2 - a2 * c1) / (b1 * a2 - b2 * a1))
    length = ((tip_point[0] - intersect_point[0]) ** 2 + (tip_point[1] - intersect_point[1]) ** 2) ** 0.5

    return tip_point, intersect_point, length


def calculate_distances(no_of_frame, contours_number, circle_x_coor, circle_y_coor, left_bound, right_bound,
                        excel_number, inner_r, outer_r, real_r, box_height, pre_width, pre_length, pre_LW_Ratio):
    # Set up parameters
    contour_areas = []
    two_tools_touch = 0
    true_mass_xs = []
    true_mass_ys = []
    number = []
    RowNumber = excel_number
    ColumnNumber = 1

    # Manually set metrics (Use Clip33 as the ref)
    # For two masks classification
    area_ratio_metrics = 10
    mask_dist_metrics = (real_r * 2) * 0.24
    end_effector_detail = (real_r * 2) * 0.18
    tool_body_concave = (real_r * 2) * 0.0449
    k_ratio = 0.3
    length_ratio_metrics = 50
    #small_area_metrics = (np.pi * true_rad * true_rad) * 0.005
    small_area_metrics = 1000

    for num_of_contours in range(len(contours_number)):
        M = cv2.moments(contours_number[num_of_contours], 0)
        if M['m00']:
            mass_centres_x = int(M['m10'] / M['m00'])
            mass_centres_y = int(M['m01'] / M['m00'])
        else:
            number.append(0)
            contour_areas.append(1)
            true_mass_xs.append(0)
            true_mass_ys.append(0)
            continue

        if mass_centres_x < left_bound:
            number.append(0)
            contour_areas.append(1)
            true_mass_xs.append(0)
            true_mass_ys.append(0)
            numbers = len(contours_number[num_of_contours])
            for i in range(numbers):
                if contours_number[num_of_contours][i][0][0] > right_bound:
                    dist = (contours_number[num_of_contours][i][0][0] - circle_x) ** 2 + (
                            contours_number[num_of_contours][i][0][1] - circle_y) ** 2
                    # If two tools contact together
                    if (inner_rad) ** 2 <= dist <= outer_rad ** 2:
                        two_tools_touch = 1
                        break

        elif left_bound <= mass_centres_x < circle_x_coor:
            right_tool = 0
            numbers = len(contours_number[num_of_contours])
            for j in range(numbers):
                if contours_number[num_of_contours][j][0][1] > 525 or contours_number[num_of_contours][j][0][1] < 15:
                    right_tool = 1

                if contours_number[num_of_contours][j][0][0] > right_bound:
                    dist = (contours_number[num_of_contours][j][0][0] - circle_x_coor) ** 2 + (
                            contours_number[num_of_contours][j][0][1] - circle_y_coor) ** 2
                    # If two tools contact together
                    if (inner_rad) ** 2 <= dist <= outer_r ** 2:
                        two_tools_touch = 1
                        break
                elif contours_number[num_of_contours][j][0][0] < left_bound:
                    dist = (contours_number[num_of_contours][j][0][0] - circle_x_coor) ** 2 + (
                            contours_number[num_of_contours][j][0][1] - circle_y_coor) ** 2
                    if (inner_rad) ** 2 <= dist <= outer_r ** 2:
                        right_tool = 0
                        break

            # Calculate min area rect
            if right_tool:
                rect = cv2.minAreaRect(contours_number[num_of_contours])
                box = cv2.boxPoints(rect)
                box_h = abs(((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1])) ** 0.5)
                box_w = abs(((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1])) ** 0.5)
                number.append(num_of_contours)
                true_mass_xs.append(mass_centres_x)
                true_mass_ys.append(mass_centres_y)

                if box_h * box_w > small_area_metrics:
                    contour_areas.append(box_h * box_w)
                else:
                    contour_areas.append(2)
            else:
                number.append(0)
                contour_areas.append(1)
                true_mass_xs.append(0)
                true_mass_ys.append(0)


        elif mass_centres_x >= circle_x_coor:
            # Calculate min area rect
            rect = cv2.minAreaRect(contours_number[num_of_contours])
            box = cv2.boxPoints(rect)
            box_h = abs(((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1])) ** 0.5)
            box_w = abs(((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1])) ** 0.5)
            if box_h * box_w > small_area_metrics:
                contour_areas.append(box_h * box_w)
            else:
                contour_areas.append(2)
            number.append(num_of_contours)
            true_mass_xs.append(mass_centres_x)
            true_mass_ys.append(mass_centres_y)

            numbers = len(contours_number[num_of_contours])
            for l in range(numbers):
                if contours_number[num_of_contours][l][0][0] < left_bound:
                    dist = (contours_number[num_of_contours][l][0][0] - circle_x_coor) ** 2 + (
                            contours_number[num_of_contours][l][0][1] - circle_y_coor) ** 2
                    if (inner_rad) ** 2 <= dist <= outer_r ** 2:
                        two_tools_touch = 1
                        break

    if two_tools_touch == 1:
        sheet.cell(row=RowNumber, column=ColumnNumber, value=('No.' + str(frames)))
        sheet.cell(row=RowNumber, column=ColumnNumber + 1, value=pre_width)
        sheet.cell(row=RowNumber, column=ColumnNumber + 2, value=pre_length)
        sheet.cell(row=RowNumber, column=ColumnNumber + 3, value=pre_LW_Ratio)
        RowNumber += 1
        cv2.putText(frame1, "Two tools contact together", (20, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
        cv2.imwrite('E:/Clip16SL/clip16' + '_' + str(frames) + '.jpg', frame1)
        print('No.' + str(frames))
        return RowNumber, pre_width, pre_length, pre_LW_Ratio

    if len(contour_areas) == 0 or max(contour_areas) < 2:
        sheet.cell(row=RowNumber, column=ColumnNumber, value=('No.' + str(frames)))
        sheet.cell(row=RowNumber, column=ColumnNumber + 1, value=0)
        sheet.cell(row=RowNumber, column=ColumnNumber + 2, value=0)
        sheet.cell(row=RowNumber, column=ColumnNumber + 3, value=0)
        RowNumber += 1
        cv2.putText(frame1, "No Tools on the right side", (20, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
        cv2.imwrite('E:/Clip16SL/clip16' + '_' + str(frames) + '.jpg', frame1)
        print('No.' + str(frames))
        return RowNumber, pre_width, pre_length, pre_LW_Ratio

    if 1 < max(contour_areas) < 3:
        sheet.cell(row=RowNumber, column=ColumnNumber, value=('No.' + str(frames)))
        sheet.cell(row=RowNumber, column=ColumnNumber + 1, value=None)
        sheet.cell(row=RowNumber, column=ColumnNumber + 2, value=None)
        sheet.cell(row=RowNumber, column=ColumnNumber + 3, value=None)
        RowNumber += 1
        cv2.putText(frame1, "right tool mask too small", (20, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
        cv2.imwrite('E:/Clip16SL/clip16' + '_' + str(frames) + '.jpg', frame1)
        print('No.' + str(frames))
        return RowNumber, pre_width, pre_length, pre_LW_Ratio

    # Find max contour on the right side
    sorted_contour_areas = sorted(contour_areas)
    max_num_id = contour_areas.index(max(contour_areas))
    #cv2.drawContours(frame1, contours_number[max_num_id], -1, (255, 0, 0), 3)
    if len(contour_areas) > 1 and sorted_contour_areas[-2] != 0:
        sec_max_num_id = contour_areas.index(sorted_contour_areas[-2])
        cv2.drawContours(frame1, contours_number[sec_max_num_id], -1, (0, 0, 255), 3)
        area_ratio = sorted_contour_areas[-2] / max(contour_areas) * 100
        cv2.putText(frame1, "Area Ratio: {:.2f}%".format(area_ratio), (20, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0))

        max_x = true_mass_xs[max_num_id]
        max_y = true_mass_ys[max_num_id]
        sec_max_x = true_mass_xs[sec_max_num_id]
        sec_max_y = true_mass_ys[sec_max_num_id]
        mask_dist = ((max_x - sec_max_x) ** 2 + (max_y - sec_max_y) ** 2) ** 0.5

        cv2.circle(frame1, (int(max_x), int(max_y)), 8, (0, 0, 255), 5)
        cv2.circle(frame1, (int(sec_max_x), int(sec_max_y)), 8, (255, 0, 0), 5)

        if area_ratio > area_ratio_metrics and mask_dist < mask_dist_metrics:
            cv2.putText(frame1, "Only have end effector", (20, 40), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
            sheet.cell(row=RowNumber, column=ColumnNumber, value=('No.' + str(frames)))
            sheet.cell(row=RowNumber, column=ColumnNumber + 1, value=None)
            sheet.cell(row=RowNumber, column=ColumnNumber + 2, value=None)
            sheet.cell(row=RowNumber, column=ColumnNumber + 3, value=None)
            RowNumber += 1
            cv2.imwrite('E:/Clip16SL/clip16' + '_' + str(frames) + '.jpg', frame1)
            print('No.' + str(frames))
            return RowNumber, pre_width, pre_length, pre_LW_Ratio


    hull = cv2.convexHull(contours_number[max_num_id], clockwise=False, returnPoints=False)
    try:
        defects = cv2.convexityDefects(contours_number[max_num_id], hull)
    except cv2.error as e:
        sheet.cell(row=RowNumber, column=ColumnNumber, value=('No.' + str(frames)))
        sheet.cell(row=RowNumber, column=ColumnNumber + 1, value=None)
        sheet.cell(row=RowNumber, column=ColumnNumber + 2, value=None)
        sheet.cell(row=RowNumber, column=ColumnNumber + 3, value=None)
        RowNumber += 1
        cv2.imwrite('E:/Clip16SL/clip16' + '_' + str(frames) + '.jpg', frame1)
        print('No.' + str(frames))
        return RowNumber, None, None, None

    far_list = []

    for j in range(defects.shape[0]):
        s, e, f, d = defects[j, 0]
        point_dist = np.int(d)
        far_list.append(point_dist)

    new_list = sorted(far_list)
    max_number = far_list.index(new_list[-1])
    s, e, f, d = defects[max_number, 0]
    far = tuple(contours_number[max_num_id][f][0])
    start = tuple(contours_number[max_num_id][s][0])
    end = tuple(contours_number[max_num_id][e][0])
    point_dist = np.int(d)
    cv2.line(frame1, start, end, (0, 225, 0), 2)
    cv2.circle(frame1, far, 8, (0, 255, 0), 5)

    cv2.putText(frame1, "Farthest dist: {:.2f}".format(point_dist / 256), (20, 60), cv2.FONT_ITALIC, 0.5,
                (0, 255, 0))

    if (point_dist / 256) > end_effector_detail:
        cv2.putText(frame1, "Longer than end_effector_detail, Dont record", (20, 80), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
        sheet.cell(row=RowNumber, column=ColumnNumber, value=('No.' + str(frames)))
        sheet.cell(row=RowNumber, column=ColumnNumber + 1, value=None)
        sheet.cell(row=RowNumber, column=ColumnNumber + 2, value=None)
        sheet.cell(row=RowNumber, column=ColumnNumber + 3, value=None)
        RowNumber += 1
        cv2.imwrite('E:/Clip16SL/clip16' + '_' + str(frames) + '.jpg', frame1)
        print('No.' + str(frames))
        return RowNumber, None, None, None

    else:
        if (point_dist / 256) > tool_body_concave:
            cv2.circle(frame1, start, 8, (255, 255, 0), 5)
            cv2.circle(frame1, end, 8, (0, 255, 0), 5)
            cv2.putText(frame1, "Consider", (20, 80), cv2.FONT_ITALIC, 0.5, (0, 255, 0))

            k_start = abs((far[1] - start[1]) / (far[0] - start[0]))
            k_end = abs((far[1] - end[1]) / (far[0] - end[0]))

            if k_start > k_end:
                if k_end / k_start < k_ratio:
                    pt1 = end
                    pt2 = far
                else:
                    pt1 = ((end[0] + start[0]) / 2, (end[1] + start[1]) / 2)
                    pt2 = far
            else:
                if k_start / k_end < k_ratio:
                    pt1 = start
                    pt2 = far
                else:
                    pt1 = ((end[0] + start[0]) / 2, (end[1] + start[1]) / 2)
                    pt2 = far
            far_to_effector = ((far[0] - pt1[0]) ** 2 + (far[1] - pt1[1]) ** 2) ** 0.5

            circle_cen = (circle_x, circle_y)
            circle_rad = true_rad
            new_point = circle_line_segment_intersection(circle_cen, circle_rad, pt1, pt2, full_line=True,
                                                         tangent_tol=1e-9)
            if new_point[0][0] > new_point[1][0]:
                new_point = (new_point[0][0], new_point[0][1])
            else:
                new_point = (new_point[1][0], new_point[1][1])
            cv2.line(frame1, far, (int(new_point[0]), int(new_point[1])), (0, 0, 225), 2)

            far_to_edge = ((far[0] - new_point[0]) ** 2 + (far[1] - new_point[1]) ** 2) ** 0.5
            length_ratio = far_to_effector / far_to_edge * 100
            cv2.putText(frame1, "effector length ratio: {:.2f}".format(length_ratio), (20, 100), cv2.FONT_ITALIC,
                        0.5,
                        (0, 255, 0))
            if length_ratio >= length_ratio_metrics:
                cv2.putText(frame1, "Longer than length_ratio_metrics, Dont record", (20, 120), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
                sheet.cell(row=RowNumber, column=ColumnNumber, value=('No.' + str(frames)))
                sheet.cell(row=RowNumber, column=ColumnNumber + 1, value=None)
                sheet.cell(row=RowNumber, column=ColumnNumber + 2, value=None)
                sheet.cell(row=RowNumber, column=ColumnNumber + 3, value=None)
                RowNumber += 1
                cv2.imwrite('E:/Clip16SL/clip16' + '_' + str(frames) + '.jpg', frame1)
                print('No.' + str(frames))
                return RowNumber, None, None, None


    cv2.putText(frame1, "Record", (20, 120), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    # Find max contour on the right side
    max_num_id = contour_areas.index(max(contour_areas))
    true_mass_x = true_mass_xs[max_num_id]
    true_mass_y = true_mass_ys[max_num_id]
    max_num = number[max_num_id]

    # Collect points from the max contour to prepare K-means
    points, curve_pathx, curve_pathy, upbox_x, upbox_y, downbox_x, downbox_y = prep_for_Kmeans(contours_number, max_num,
                                                                                               circle_x_coor,
                                                                                               circle_y_coor,
                                                                                               left_bound,
                                                                                               right_bound, inner_r,
                                                                                               outer_r, box_height)
    # Run the K means to get feature points
    centers, have_centers = kmeans_algorithm(points)
    if have_centers == 0:
        sheet.cell(row=RowNumber, column=ColumnNumber, value=('No.' + str(frames)))
        sheet.cell(row=RowNumber, column=ColumnNumber + 1, value=0)
        sheet.cell(row=RowNumber, column=ColumnNumber + 2, value=0)
        sheet.cell(row=RowNumber, column=ColumnNumber + 3, value=0)
        RowNumber += 1
        return RowNumber, pre_width, pre_length, pre_LW_Ratio
    for l in range(len(centers)):
        cv2.circle(frame1, (int(centers[l][0]), int(centers[l][1])), radius=3, color=(0, 255, 0), thickness=-1)

    # Locating the mask by using three points (two edges and one tip)
    p4 = (true_mass_x, true_mass_y)

    if len(downbox_x) and len(curve_pathx):
        pr, pd = get_feature_points_b(centers, bottom_or_top=1)
        pnr = get_next_point(centers, pr, add_x_restriction=1)
        pnd = get_next_point(centers, pd, add_x_restriction=0)
        k5, p5 = find_angle_bisector_curveedge(pr, pnr, pd, pnd, centers)
        p3_index = curve_pathy.index(min(curve_pathy))
        p3 = (curve_pathx[p3_index], curve_pathy[p3_index])
        p2 = get_symmetric_point_b(p5, p3, k5)

        cv2.line(frame1, (int(pr[0]), int(pr[1])), (int(pnr[0]), int(pnr[1])), (255, 0, 0), thickness=5)
        cv2.line(frame1, (int(pd[0]), int(pd[1])), (int(pnd[0]), int(pnd[1])), (255, 0, 0), thickness=5)

    elif len(downbox_x) and len(curve_pathx) == 0:
        pr, pnr, pl, pnl = get_feature_points_c(centers, right_bound, left_bound, bottom_or_top=1)
        k5, p5 = find_angle_bisector(pr, pnr, pl, pnl, centers)
        p2_index = downbox_x.index(min(downbox_x))
        p2 = (downbox_x[p2_index], downbox_y[p2_index])
        p3_index = downbox_x.index(max(downbox_x))
        p3 = (downbox_x[p3_index], downbox_y[p3_index])
        p2, p3 = get_symmetric_point_a(p5, p3, p2, k5)

        cv2.line(frame1, (int(pr[0]), int(pr[1])), (int(pnr[0]), int(pnr[1])), (255, 0, 0), thickness=5)
        cv2.line(frame1, (int(pl[0]), int(pl[1])), (int(pnl[0]), int(pnl[1])), (255, 0, 0), thickness=5)

    elif len(upbox_x) and len(curve_pathx):
        pr, pu = get_feature_points_b(centers, bottom_or_top=0)
        pnr = get_next_point(centers, pr, add_x_restriction=1)
        pnu = get_next_point(centers, pu, add_x_restriction=0)
        k5, p5 = find_angle_bisector_curveedge(pu, pnu, pr, pnr, centers)
        p3_index = curve_pathy.index(max(curve_pathy))
        p3 = (curve_pathx[p3_index], curve_pathy[p3_index])
        p2 = get_symmetric_point_b(p5, p3, k5)

        cv2.line(frame1, (int(pr[0]), int(pr[1])), (int(pnr[0]), int(pnr[1])), (255, 0, 0), thickness=5)
        cv2.line(frame1, (int(pu[0]), int(pu[1])), (int(pnu[0]), int(pnu[1])), (255, 0, 0), thickness=5)

    elif len(upbox_x) and len(curve_pathx) == 0:
        pl, pnl, pr, pnr = get_feature_points_c(centers, right_bound, left_bound, bottom_or_top=0)
        k5, p5 = find_angle_bisector(pl, pnl, pr, pnr, centers)
        p2_index = upbox_x.index(min(upbox_x))
        p2 = (upbox_x[p2_index], upbox_y[p2_index])
        p3_index = upbox_x.index(max(upbox_x))
        p3 = (upbox_x[p3_index], upbox_y[p3_index])
        p2, p3 = get_symmetric_point_a(p5, p3, p2, k5)

        cv2.line(frame1, (int(pl[0]), int(pl[1])), (int(pnl[0]), int(pnl[1])), (255, 0, 0), thickness=5)
        cv2.line(frame1, (int(pr[0]), int(pr[1])), (int(pnr[0]), int(pnr[1])), (255, 0, 0), thickness=5)

    elif len(curve_pathx):
        pu, pnu, pd, pnd = get_feature_points_a(centers, real_r, circle_x_coor, circle_y_coor)
        k5, p5 = find_angle_bisector_curve(pu, pnu, pd, pnd, centers)
        p2_index = curve_pathy.index(min(curve_pathy))
        p2 = (curve_pathx[p2_index], curve_pathy[p2_index])
        p3_index = curve_pathy.index(max(curve_pathy))
        p3 = (curve_pathx[p3_index], curve_pathy[p3_index])
        p2, p3 = get_symmetric_point_a(p5, p3, p2, k5)

        cv2.line(frame1, (int(pd[0]), int(pd[1])), (int(pnd[0]), int(pnd[1])), (255, 0, 0), thickness=5)
        cv2.line(frame1, (int(pu[0]), int(pu[1])), (int(pnu[0]), int(pnu[1])), (0, 0, 255), thickness=5)

    else:
        sheet.cell(row=RowNumber, column=ColumnNumber, value=('No.' + str(frames)))
        sheet.cell(row=RowNumber, column=ColumnNumber + 1, value=None)
        sheet.cell(row=RowNumber, column=ColumnNumber + 2, value=None)
        sheet.cell(row=RowNumber, column=ColumnNumber + 3, value=None)
        cv2.imwrite('E:/Clip16SL/clip16' + '_' + str(frames) + '.jpg', frame1)
        print('No.' + str(frames))
        RowNumber += 1
        return RowNumber, pre_width, pre_length, pre_LW_Ratio

    pointsdist = ((p2[1] - p3[1]) ** 2 + (p2[0] - p3[0]) ** 2) ** 0.5
    arclength = asin(circle_y / real_r) * 2 * real_r
    pointsdist = pointsdist / arclength * 100
    pre_width = pointsdist

    p_tip1, p_tip2, length = get_tool_length(p2, p3, centers, k5)
    length = length / (real_r * 2) * 100
    pre_length = length
    if pointsdist != 0:
        LW_Ratio = length / pointsdist
        if LW_Ratio > 20:
            LW_Ratio = 0
    else:
        LW_Ratio = 0
    pre_LW_Ratio = LW_Ratio

    sheet.cell(row=RowNumber, column=ColumnNumber, value=('No.' + str(frames)))
    sheet.cell(row=RowNumber, column=ColumnNumber + 1, value=pointsdist)
    sheet.cell(row=RowNumber, column=ColumnNumber + 2, value=length)
    sheet.cell(row=RowNumber, column=ColumnNumber + 3, value=LW_Ratio)
    RowNumber += 1

    cv2.line(frame1, (int(p5[0]), int(p5[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), thickness=1)
    cv2.line(frame1, (int(p5[0]), int(p5[1])), (int(p3[0]), int(p3[1])), (0, 0, 255), thickness=1)
    cv2.line(frame1, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (255, 0, 0), thickness=2)
    cv2.line(frame1, (int(p5[0]), int(p5[1])), (int(p5[0]) + 900, int(p5[1] + 900 * k5)), (255, 0, 0), thickness=2)
    cv2.line(frame1, (int(p_tip1[0]), int(p_tip1[1])), (int(p_tip2[0]), int(p_tip2[1])), (0, 255, 0), thickness=2)

    cv2.imwrite('E:/Clip16SL/clip16' + '_' + str(frames) + '.jpg', frame1)
    print('No.' + str(frames))
    cv2.waitKey(0)

    return RowNumber, pre_width, pre_length, pre_LW_Ratio


if __name__ == "__main__":
    # Excel setup
    tablepath = 'Clip16_AngleTest5120-5640SL.xlsx'
    Row_Number = 2
    ColumnNumber = 1
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.cell(row=1, column=ColumnNumber, value='Img')
    sheet.cell(row=1, column=ColumnNumber + 1, value='Width')
    sheet.cell(row=1, column=ColumnNumber + 2, value='Length')
    sheet.cell(row=1, column=ColumnNumber + 3, value='LW_Ratio')
    prewidth = 0
    prelength = 0
    prelwratio = 0
    # ---------------------------------

    # Laparoscopic view geo parameters
    video_num = 78
    video_constant = constant.VideoConstants()
    constant_values = video_constant.num_to_constants(video_num)()

    true_rad = constant_values[0]
    inner_rad = constant_values[1]
    outer_rad = constant_values[2]
    circle_x = constant_values[3]
    circle_y = constant_values[4]
    boundaryl = constant_values[5]
    boundaryr = constant_values[6]
    boxheight = 10
    # --------------------------------

    for frames in range(2, 832, 1):
        if video_num % 10 == 0:
            folder_name = str(10 * (video_num // 10) - 9) + '-' + str(10 * (video_num // 10))
        else:
            folder_name = str(10*(video_num//10)+1) + '-' + str(10*(video_num//10)+10)
        frame = cv2.imread('E:/Clip' + folder_name + '/Clip' + str(video_num) + '_1M/clip_' + str(video_num) + '' + str(
            frames) + 'M.jpg')
        frame1 = cv2.imread(
            'E:/Clip' + folder_name + '/Clip' + str(video_num) + '_1D/clip' + str(video_num) + '_' + str(
                frames) + 'D.jpg')
        mask_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, contours_frame = cv2.threshold(mask_frame, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(contours_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Row_Number, prewidth, prelength, prelwratio = calculate_distances(frames, contours, circle_x, circle_y,
                                                                          boundaryl, boundaryr,
                                                                          Row_Number, inner_rad, outer_rad, true_rad,
                                                                          boxheight, prewidth, prelength, prelwratio)

    workbook.save(tablepath)
    print('Finish')
