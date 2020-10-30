from math import atan, tan, asin, pi, atan2

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
    # Prevent the two edges share the same point
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

def get_feature_points_a_use_k(centers_matrix, circle_rad, circle_x_coor, circle_y_coor):
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

    max_point = (centers_matrix[max_number][0], centers_matrix[max_number][1])
    sec_max_point = (centers_matrix[sec_max_number][0], centers_matrix[sec_max_number][1])
    mid_point = ((max_point[0]+sec_max_point[0])/2, (max_point[1]+sec_max_point[1])/2)

    base_vector = ((sec_max_point[0]-max_point[0]), (sec_max_point[1]-max_point[1]))

    list_interval_angle = []
    for i in range(len(centers_matrix)):
        if i == max_number or i == sec_max_number:
            list_interval_angle.append(0)
            continue
        dx1 = base_vector[0]
        dy1 = base_vector[1]
        dx2 = centers_matrix[i][0] - mid_point[0]
        dy2 = centers_matrix[i][1] - mid_point[1]
        angle1 = atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / pi)
        angle2 = atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / pi)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
        list_interval_angle.append(included_angle)
    sorted_list_interval_angle = sorted(list_interval_angle)
    next_max_number = list_interval_angle.index(sorted_list_interval_angle[-1])
    next_sec_max_number = list_interval_angle.index(sorted_list_interval_angle[2])

    if centers_matrix[max_number][1] < centers_matrix[sec_max_number][1]:
        right_top_point = (centers_matrix[max_number][0], centers_matrix[max_number][1])
        next_right_top_point = (centers_matrix[next_max_number][0], centers_matrix[next_max_number][1])
        right_bottom_point = (centers_matrix[sec_max_number][0], centers_matrix[sec_max_number][1])
        next_right_bottom_point = (centers_matrix[next_sec_max_number][0], centers_matrix[next_sec_max_number][1])
    else:
        right_top_point = (centers_matrix[sec_max_number][0], centers_matrix[sec_max_number][1])
        next_right_top_point = (centers_matrix[next_sec_max_number][0], centers_matrix[next_sec_max_number][1])
        right_bottom_point = (centers_matrix[max_number][0], centers_matrix[max_number][1])
        next_right_bottom_point = (centers_matrix[next_max_number][0], centers_matrix[next_max_number][1])

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