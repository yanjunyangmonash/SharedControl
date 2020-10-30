

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