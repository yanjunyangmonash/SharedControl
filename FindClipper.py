import cv2
import openpyxl
from math import atan2, asin, pi
import numpy as np
import constant
from utils import GeoCalculation as GC
from utils import kmeans as k
from utils import symmetricpoints as sp
from utils import featurepoints as fp
from utils import anglebisector as ab


def calculate_distances(contours_number, excel_number, pre_width, pre_length, pre_LW_Ratio):
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
    area_ratio_metrics = 40
    mask_dist_metrics = (true_rad * 2) * 0.3
    end_effector_detail = (true_rad * 2) * 0.18
    tool_body_concave = (true_rad * 2) * 0.0449
    k_ratio = 0.3
    length_ratio_metrics = 50
    angle = atan2(270, (boundaryr - circle_x))
    angle = int(angle * 180 / pi)
    lapview_area = 270 * (boundaryr - boundaryl) + (pi * (true_rad ** 2)) * (angle / 180)
    small_area_metrics = lapview_area * 0.03
    big_area_metrics = small_area_metrics * 3

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

        if mass_centres_x < boundaryl:
            number.append(0)
            contour_areas.append(1)
            true_mass_xs.append(0)
            true_mass_ys.append(0)
            numbers = len(contours_number[num_of_contours])
            for i in range(numbers):
                if contours_number[num_of_contours][i][0][0] > boundaryr:
                    dist = (contours_number[num_of_contours][i][0][0] - circle_x) ** 2 + (
                            contours_number[num_of_contours][i][0][1] - circle_y) ** 2
                    # If two tools contact together
                    if (inner_rad) ** 2 <= dist <= outer_rad ** 2:
                        two_tools_touch = 1
                        break

        # This part can be modified to speed up the process
        elif boundaryl <= mass_centres_x < circle_x:
            right_tool = 0
            no_left_tool = 1
            numbers = len(contours_number[num_of_contours])
            for j in range(numbers):
                if contours_number[num_of_contours][j][0][1] > 525 or contours_number[num_of_contours][j][0][1] < 15:
                    right_tool = 1

                if contours_number[num_of_contours][j][0][0] > boundaryr:
                    dist = (contours_number[num_of_contours][j][0][0] - circle_x) ** 2 + (
                            contours_number[num_of_contours][j][0][1] - circle_y) ** 2
                    # If two tools contact together
                    if inner_rad ** 2 <= dist <= outer_rad ** 2:
                        two_tools_touch = 1
                        #break
                elif (contours_number[num_of_contours][j][0][0] > circle_x * 1.2 and
                        contours_number[num_of_contours][j][0][1] > 525) or (
                        contours_number[num_of_contours][j][0][0] > circle_x * 1.2 and
                        contours_number[num_of_contours][j][0][1] < 15):
                    two_tools_touch = 1
                    #break
                elif contours_number[num_of_contours][j][0][0] < boundaryl:
                    if no_left_tool == 1:
                        dist = (contours_number[num_of_contours][j][0][0] - circle_x) ** 2 + (
                                contours_number[num_of_contours][j][0][1] - circle_y) ** 2
                        if inner_rad ** 2 <= dist <= outer_rad ** 2:
                            no_left_tool = 0
                            #break
                two_tools_touch = two_tools_touch - no_left_tool

            # Calculate min area rect
            if right_tool * no_left_tool > 0:
                rect = cv2.minAreaRect(contours_number[num_of_contours])
                box = cv2.boxPoints(rect)
                box_h = abs(((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2) ** 0.5)
                box_w = abs(((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2) ** 0.5)
                box = np.int0(box)
                cv2.drawContours(frame1, [box], 0, (0, 255, 255), 3)
                number.append(num_of_contours)
                true_mass_xs.append(mass_centres_x)
                true_mass_ys.append(mass_centres_y)

                #if box_h * box_w > small_area_metrics:
                contour_areas.append(box_h * box_w)
                #else:
                    #contour_areas.append(2)
            else:
                number.append(0)
                contour_areas.append(1)
                true_mass_xs.append(0)
                true_mass_ys.append(0)


        elif mass_centres_x >= circle_x:
            # Calculate min area rect
            rect = cv2.minAreaRect(contours_number[num_of_contours])
            box = cv2.boxPoints(rect)
            box_h = abs(((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2) ** 0.5)
            box_w = abs(((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2) ** 0.5)
            #if box_h * box_w > small_area_metrics:
            contour_areas.append(box_h * box_w)
            box = np.int0(box)
            cv2.drawContours(frame1, [box], 0, (0, 255, 255), 3)
            #else:
                #contour_areas.append(2)
            number.append(num_of_contours)
            true_mass_xs.append(mass_centres_x)
            true_mass_ys.append(mass_centres_y)

            numbers = len(contours_number[num_of_contours])
            for l in range(numbers):
                if contours_number[num_of_contours][l][0][0] < boundaryl:
                    dist = (contours_number[num_of_contours][l][0][0] - circle_x) ** 2 + (
                            contours_number[num_of_contours][l][0][1] - circle_y) ** 2
                    if (inner_rad) ** 2 <= dist <= outer_rad ** 2:
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

    # Find max contour on the right side
    sorted_contour_areas = sorted(contour_areas)
    # Use position relationship to make sure the algorithm always tracks the correct tool (Confirm with Arvind!!!)
    max_num_id = contour_areas.index(max(contour_areas))
    if len(contour_areas) > 1 and sorted_contour_areas[-2] > small_area_metrics/5:
        sec_max_num_id = contour_areas.index(sorted_contour_areas[-2])
        if (true_mass_xs[max_num_id] - circle_x)*(true_mass_xs[sec_max_num_id] - circle_x) > 0:
            if true_mass_xs[max_num_id] * true_mass_ys[max_num_id] < true_mass_xs[sec_max_num_id] * true_mass_ys[
                    sec_max_num_id]:
                max_num_id = contour_areas.index(sorted_contour_areas[-2])
                sec_max_num_id = contour_areas.index(sorted_contour_areas[-1])

        else:
            if true_mass_xs[max_num_id] < true_mass_xs[sec_max_num_id]:
                max_num_id = contour_areas.index(sorted_contour_areas[-2])
                sec_max_num_id = contour_areas.index(sorted_contour_areas[-1])

        # If the mask area is big enough, it shouldn't be considered as an end-effector's mask
        if sorted_contour_areas[-1] < big_area_metrics:
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

    if max(contour_areas) < small_area_metrics:
        sheet.cell(row=RowNumber, column=ColumnNumber, value=('No.' + str(frames)))
        sheet.cell(row=RowNumber, column=ColumnNumber + 1, value=None)
        sheet.cell(row=RowNumber, column=ColumnNumber + 2, value=None)
        sheet.cell(row=RowNumber, column=ColumnNumber + 3, value=None)
        RowNumber += 1
        cv2.putText(frame1, "right tool mask too small", (20, 40), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
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
            new_point = GC.circle_line_segment_intersection(circle_cen, circle_rad, pt1, pt2, full_line=True,
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
                cv2.putText(frame1, "Longer than length_ratio_metrics, Dont record", (20, 120), cv2.FONT_ITALIC, 0.5,
                            (0, 255, 0))
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
    true_mass_x = true_mass_xs[max_num_id]
    true_mass_y = true_mass_ys[max_num_id]
    max_num = number[max_num_id]

    # Collect points from the max contour to prepare K-means
    points, curve_pathx, curve_pathy, upbox_x, upbox_y, downbox_x, downbox_y = k.prep_for_Kmeans(contours_number,
                                                                                                 max_num,
                                                                                                 circle_x,
                                                                                                 circle_y,
                                                                                                 boundaryl,
                                                                                                 boundaryr, inner_rad,
                                                                                                 outer_rad, boxheight)
    # Run the K means to get feature points
    centers, have_centers = k.kmeans_algorithm(points)
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
        # pr, pd = fp.get_feature_points_b(centers, bottom_or_top=1)
        # pnr = fp.get_next_point(centers, pr, add_x_restriction=1)
        # pnd = fp.get_next_point(centers, pd, add_x_restriction=0)
        pr, pnr, pd, pnd = fp.get_feature_points_b_use_k(centers, bottom_or_top=1)
        k5, p5 = ab.find_angle_bisector_curveedge(pr, pnr, pd, pnd, centers)
        p3_index = curve_pathy.index(min(curve_pathy))
        p3 = (curve_pathx[p3_index], curve_pathy[p3_index])
        p2 = sp.get_symmetric_point_b(p5, p3, k5)

        cv2.line(frame1, (int(pr[0]), int(pr[1])), (int(pnr[0]), int(pnr[1])), (255, 0, 0), thickness=5)
        cv2.line(frame1, (int(pd[0]), int(pd[1])), (int(pnd[0]), int(pnd[1])), (255, 0, 0), thickness=5)

    elif len(downbox_x) and len(curve_pathx) == 0:
        pr, pnr, pl, pnl = fp.get_feature_points_c_use_k(centers, bottom_or_top=1)
        k5, p5 = ab.find_angle_bisector(pr, pnr, pl, pnl, centers)
        p2_index = downbox_x.index(min(downbox_x))
        p2 = (downbox_x[p2_index], downbox_y[p2_index])
        p3_index = downbox_x.index(max(downbox_x))
        p3 = (downbox_x[p3_index], downbox_y[p3_index])
        p2, p3 = sp.get_symmetric_point_a(p5, p3, p2, k5)

        cv2.line(frame1, (int(pr[0]), int(pr[1])), (int(pnr[0]), int(pnr[1])), (255, 0, 0), thickness=5)
        cv2.line(frame1, (int(pl[0]), int(pl[1])), (int(pnl[0]), int(pnl[1])), (255, 0, 0), thickness=5)

    elif len(upbox_x) and len(curve_pathx):
        # pr, pu = fp.get_feature_points_b(centers, bottom_or_top=0)
        # pnr = fp.get_next_point(centers, pr, add_x_restriction=1)
        # pnu = fp.get_next_point(centers, pu, add_x_restriction=0)
        pr, pnr, pu, pnu = fp.get_feature_points_b_use_k(centers, bottom_or_top=0)
        k5, p5 = ab.find_angle_bisector_curveedge(pu, pnu, pr, pnr, centers)
        p3_index = curve_pathy.index(max(curve_pathy))
        p3 = (curve_pathx[p3_index], curve_pathy[p3_index])
        p2 = sp.get_symmetric_point_b(p5, p3, k5)

        cv2.line(frame1, (int(pr[0]), int(pr[1])), (int(pnr[0]), int(pnr[1])), (255, 0, 0), thickness=5)
        cv2.line(frame1, (int(pu[0]), int(pu[1])), (int(pnu[0]), int(pnu[1])), (255, 0, 0), thickness=5)

    elif len(upbox_x) and len(curve_pathx) == 0:
        pl, pnl, pr, pnr = fp.get_feature_points_c_use_k(centers, bottom_or_top=0)
        k5, p5 = ab.find_angle_bisector(pl, pnl, pr, pnr, centers)
        p2_index = upbox_x.index(min(upbox_x))
        p2 = (upbox_x[p2_index], upbox_y[p2_index])
        p3_index = upbox_x.index(max(upbox_x))
        p3 = (upbox_x[p3_index], upbox_y[p3_index])
        p2, p3 = sp.get_symmetric_point_a(p5, p3, p2, k5)

        cv2.line(frame1, (int(pl[0]), int(pl[1])), (int(pnl[0]), int(pnl[1])), (255, 0, 0), thickness=5)
        cv2.line(frame1, (int(pr[0]), int(pr[1])), (int(pnr[0]), int(pnr[1])), (255, 0, 0), thickness=5)

    elif len(curve_pathx):
        pu, pnu, pd, pnd = fp.get_feature_points_a_use_k(centers, true_rad, circle_x, circle_y)
        k5, p5 = ab.find_angle_bisector_curve(pu, pnu, pd, pnd, centers)
        p2_index = curve_pathy.index(min(curve_pathy))
        p2 = (curve_pathx[p2_index], curve_pathy[p2_index])
        p3_index = curve_pathy.index(max(curve_pathy))
        p3 = (curve_pathx[p3_index], curve_pathy[p3_index])
        p2, p3 = sp.get_symmetric_point_a(p5, p3, p2, k5)

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
    arclength = asin(circle_y / true_rad) * 2 * true_rad
    pointsdist = pointsdist / arclength * 100
    pre_width = pointsdist

    p_tip1, p_tip2, length = GC.get_tool_length(p2, p3, centers, k5)
    length = length / (true_rad * 2) * 100
    pre_length = length
    if pointsdist != 0:
        LW_Ratio = length / pointsdist
        if LW_Ratio > 4 or LW_Ratio < 0.5:
            cv2.putText(frame1, "Strange data", (20, 80), cv2.FONT_ITALIC, 0.5,
                        (0, 255, 0))
            sheet.cell(row=RowNumber, column=ColumnNumber, value=('No.' + str(frames)))
            sheet.cell(row=RowNumber, column=ColumnNumber + 1, value=None)
            sheet.cell(row=RowNumber, column=ColumnNumber + 2, value=None)
            sheet.cell(row=RowNumber, column=ColumnNumber + 3, value=None)
            RowNumber += 1
            cv2.imwrite('E:/Clip16SL/clip16' + '_' + str(frames) + '.jpg', frame1)
            print('No.' + str(frames))
            return RowNumber, None, None, None
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
    video_num = 32
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

    for frames in range(1337, 2322, 1):
        if video_num % 10 == 0:
            folder_name = str(10 * (video_num // 10) - 9) + '-' + str(10 * (video_num // 10))
        else:
            folder_name = str(10 * (video_num // 10) + 1) + '-' + str(10 * (video_num // 10) + 10)
        frame = cv2.imread('E:/Clip' + folder_name + '/Clip' + str(video_num) + '_1M/clip' + str(video_num) + '_' + str(
            frames) + 'M.jpg')
        frame1 = cv2.imread(
            'E:/Clip' + folder_name + '/Clip' + str(video_num) + '_1D/clip' + str(video_num) + '_' + str(
                frames) + 'D.jpg')
        mask_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, contours_frame = cv2.threshold(mask_frame, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(contours_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Row_Number, prewidth, prelength, prelwratio = calculate_distances(contours, Row_Number, prewidth, prelength,
                                                                          prelwratio)

    workbook.save(tablepath)
    print('Finish')
