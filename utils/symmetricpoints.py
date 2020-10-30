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