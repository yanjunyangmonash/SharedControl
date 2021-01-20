import constant
from math import atan2, asin, pi
import cv2
import numpy as np
from utils import write_excel as we
from utils import GeoCalculation as GC
from FindClipper import video_num


# Laparoscopic view geo parameters
video_num = video_num
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


def valid_mask_collector(contours_number):
    # This function is used to classify masks in a image into primary tool, auxiliary tool, touched tools and no tool
    # Set up parameters
    contour_areas = []
    true_mass_xs = []
    true_mass_ys = []
    number = []

    def tool_at_the_edge(tool_on_the_right=1):
        tools_touch = 0
        if tool_on_the_right:
            for i in range(numbers):
                if contours_number[num_of_contours][i][0][0] < boundaryl:
                    dist = (contours_number[num_of_contours][i][0][0] - circle_x) ** 2 + (
                            contours_number[num_of_contours][i][0][1] - circle_y) ** 2
                    # If two tools contact together
                    if (inner_rad) ** 2 <= dist <= outer_rad ** 2:
                        tools_touch = 1
        else:
            for i in range(numbers):
                if contours_number[num_of_contours][i][0][0] > boundaryr:
                    dist = (contours_number[num_of_contours][i][0][0] - circle_x) ** 2 + (
                            contours_number[num_of_contours][i][0][1] - circle_y) ** 2
                    # If two tools contact together
                    if (inner_rad) ** 2 <= dist <= outer_rad ** 2:
                        tools_touch = 1
        return tools_touch

    def tool_on_the_left():
        tools_touch = 0
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
                    tools_touch = 1

            elif (contours_number[num_of_contours][j][0][0] > circle_x * 1.2 and
                  contours_number[num_of_contours][j][0][1] > 525) or (
                    contours_number[num_of_contours][j][0][0] > circle_x * 1.2 and
                    contours_number[num_of_contours][j][0][1] < 15):
                tools_touch = 1

            elif contours_number[num_of_contours][j][0][0] < boundaryl:
                if no_left_tool == 1:
                    dist = (contours_number[num_of_contours][j][0][0] - circle_x) ** 2 + (
                            contours_number[num_of_contours][j][0][1] - circle_y) ** 2
                    if inner_rad ** 2 <= dist <= outer_rad ** 2:
                        no_left_tool = 0

            tools_touch = tools_touch - no_left_tool

        # Calculate min area rect
        if right_tool * no_left_tool > 0:
            tool_rec_area()

        else:
            number.append(0)
            contour_areas.append(1)
            true_mass_xs.append(0)
            true_mass_ys.append(0)
        return tools_touch

    def tool_rec_area():
        # Calculate the mask's rough area, cv2.contourArea has error
        rect = cv2.minAreaRect(contours_number[num_of_contours])
        box_size = cv2.boxPoints(rect)
        box_h = abs(((box_size[0][0] - box_size[1][0]) ** 2 + (box_size[0][1] - box_size[1][1]) ** 2) ** 0.5)
        box_w = abs(((box_size[1][0] - box_size[2][0]) ** 2 + (box_size[1][1] - box_size[2][1]) ** 2) ** 0.5)
        # box_size = np.int0(box_size)
        # cv2.drawContours(frame1, [box_size], 0, (0, 255, 255), 3)
        number.append(num_of_contours)
        true_mass_xs.append(mass_centres_x)
        true_mass_ys.append(mass_centres_y)
        contour_areas.append(box_h * box_w)

    # If a mask can be recognized, the mass center will be recorded
    two_tools_touch = 0
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
            two_tools_touch = tool_at_the_edge(0)
            if two_tools_touch:
                break

        # This part can be modified to speed up the process
        elif boundaryl <= mass_centres_x < circle_x:
            two_tools_touch = tool_on_the_left()


        elif mass_centres_x >= circle_x:
            # Calculate min area rect
            tool_rec_area()
            numbers = len(contours_number[num_of_contours])
            two_tools_touch = tool_at_the_edge()
            if two_tools_touch:
                break

    return two_tools_touch, contour_areas, true_mass_xs, true_mass_ys, number


def touched_tools_filter(row_num, frame_num, pre_width, pre_length, pre_LW_Ratio, workbook, frame_mask):
    # Filter out touched tools mask
    #we.write_excel_table(frame_num, workbook, row_num, pre_width, pre_length, pre_LW_Ratio)
    we.write_excel_table(frame_num, workbook, row_num)
    cv2.putText(frame_mask, "Two tools contact together", (20, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    cv2.imwrite('C:/D/Clip16SL/clip16' + '_' + str(frame_num) + '.jpg', frame_mask)
    print('No.' + str(frame_num))


def left_tool_filter(row_num, frame_num, workbook, frame_mask):
    # Filter out auxiliary tool mask
    we.write_excel_table(frame_num, workbook, row_num)
    cv2.putText(frame_mask, "No Tools on the right side", (20, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    cv2.imwrite('C:/D/Clip16SL/clip16' + '_' + str(frame_num) + '.jpg', frame_mask)
    print('No.' + str(frame_num))


def noise_filter(frame_num, frame_mask, workbook, row_num):
    # Filter out very small masks
    we.write_excel_table(frame_num, workbook, row_num)
    cv2.putText(frame_mask, "right tool mask too small", (20, 40), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    cv2.imwrite('C:/D/Clip16SL/clip16' + '_' + str(frame_num) + '.jpg', frame_mask)
    print('No.' + str(frame_num))


def end_effector_filter(frame_num, frame_mask, contour_area_sets, mass_xs, mass_ys, workbook, row_num):
    def get_potential_ee(dist1, dist2=0, three_tools=1):
        if three_tools:
            if dist1 > dist2:
                thr_max_id = contour_area_sets.index(sorted_contour_areas[-3])
                area_ratio = sorted_contour_areas[-3] / sorted_contour_areas[-2] * 100
                max_x = mass_xs[sec_max_num_id]
                max_y = mass_ys[sec_max_num_id]
                sec_max_x = mass_xs[thr_max_id]
                sec_max_y = mass_ys[thr_max_id]
                mask_dist = dist2
                return [area_ratio, max_x, max_y, sec_max_x, sec_max_y, mask_dist]
        else:
            area_ratio = sorted_contour_areas[-2] / sorted_contour_areas[-1] * 100
            max_x = mass_xs[max_num_id]
            max_y = mass_ys[max_num_id]
            sec_max_x = mass_xs[sec_max_num_id]
            sec_max_y = mass_ys[sec_max_num_id]
            mask_dist = dist1
            return [area_ratio, max_x, max_y, sec_max_x, sec_max_y, mask_dist]

    def find_two_max_masks():
        max_x = mass_xs[max_num_id]
        max_y = mass_ys[max_num_id]
        sec_max_x = mass_xs[sec_max_num_id]
        sec_max_y = mass_ys[sec_max_num_id]
        mask_dist1 = ((max_x - sec_max_x) ** 2 + (max_y - sec_max_y) ** 2) ** 0.5
        if len(sorted_contour_areas) > 2 and sorted_contour_areas[-3] > small_area_metrics:
            thr_max_num_id = contour_area_sets.index(sorted_contour_areas[-3])
            thr_max_x = mass_xs[thr_max_num_id]
            thr_max_y = mass_ys[thr_max_num_id]
            mask_dist2 = ((sec_max_x - thr_max_x) ** 2 + (sec_max_y - thr_max_y) ** 2) ** 0.5
            masks_geometric_relation = get_potential_ee(mask_dist1, mask_dist2, three_tools=1)
        else:
            masks_geometric_relation = get_potential_ee(mask_dist1, three_tools=0)
        return masks_geometric_relation

    sorted_contour_areas = sorted(contour_area_sets)
    max_num_id = contour_area_sets.index(max(contour_area_sets))
    sec_max_num_id = 500
    only_ee = 0
    # If there are multiple masks, and three max masks are not noise, we compare the closest two masks among three
    # to decide whether these are end effectors or two different tools
    # small_area_metrics is the min value to be not considered as noise
    if len(contour_area_sets) > 1 and sorted_contour_areas[-2] > small_area_metrics:
        sec_max_num_id = contour_area_sets.index(sorted_contour_areas[-2])
        # big_area_metrics is the min value to be not considered as an end-effector's mask
        if sorted_contour_areas[-1] < big_area_metrics:
            returned_values = find_two_max_masks()

            # Draw notes on the image
            cv2.putText(frame_mask, "Area Ratio: {:.2f}%".format(returned_values[0]), (20, 20), cv2.FONT_ITALIC, 0.5,
                        (0, 255, 0))
            cv2.circle(frame_mask, (int(returned_values[1]), int(returned_values[2])), 8, (0, 0, 255), 5)
            cv2.circle(frame_mask, (int(returned_values[3]), int(returned_values[4])), 8, (255, 0, 0), 5)

            # If selected two masks has similar size and very close, assume mask only has ee part
            if returned_values[0] > area_ratio_metrics and returned_values[5] < mask_dist_metrics:
                cv2.putText(frame_mask, "Only have end effector", (20, 40), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
                we.write_excel_table(frame_num, workbook, row_num)
                cv2.imwrite('C:/D/Clip16SL/clip16' + '_' + str(frame_num) + '.jpg', frame_mask)
                print('No.' + str(frame_num))
                only_ee = 1
                return only_ee, None, None
    return only_ee, max_num_id, sec_max_num_id


def main_tool_tracker(row_num, frame_num, pre_width, workbook, frame_mask, has_main_tool, contour_area_sets, mass_xs,
                      mass_ys, max_num_id, sec_max_num_id, main_coor, assist_coor):
    # This func is used to track the primary tool
    # includes the situation when it rejoins the view and has a smaller area
    def cal_tools_move_dist(t1_x, t1_y, t2):
        tool_move_dist = ((t1_x - t2[0]) ** 2 + (t1_y - t2[1]) ** 2) ** 0.5
        return tool_move_dist

    def save_no_main_tool_data(text):
        we.write_excel_table(frame_num, workbook, row_num)
        cv2.putText(frame_mask, text, (20, 80), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
        cv2.imwrite('C:/D/Clip16SL/clip16' + '_' + str(frame_num) + '.jpg', frame_mask)
        print('No.' + str(frame_num))

    def locate_new_main_tool():
        has_main_tool = 0
        main_tool_coor = main_coor
        if pre_width != 0 or pre_width is not None:
        # Use x-axis position to classify the main tool and assist tool
            if mass_xs[max_num_id] > circle_x-(circle_x-boundaryl)/2:
                has_main_tool = 1
                main_tool_coor = (mass_xs[max_num_id], mass_ys[max_num_id])
            else:
                save_no_main_tool_data("Main tool is not on the right side")
        return has_main_tool, main_tool_coor

    def current_tool_coor():
        assist_tool_coor = (0, 0)
        main_tool_coor = main_coor
        new_max_num_id = max_num_id
        track_main_tool = 1
        # If the assist tool has the max area in the last frame
        if assist_coor[0] != 0:
            # Determine if the assist tool is still the max mask in the view
            assist_tool_move_dist = cal_tools_move_dist(mass_xs[max_num_id], mass_ys[max_num_id], assist_coor)
            if assist_tool_move_dist > mask_dist_metrics:
                main_tool_coor = (mass_xs[max_num_id], mass_ys[max_num_id])
            else:
                if len(contour_area_sets) > 1 and sorted_contour_areas[-2] > small_area_metrics / 5:
                    # Use main tool move distance but no original assist tool move distance any more
                    main_tool_movement = cal_tools_move_dist(mass_xs[sec_max_num_id], mass_ys[sec_max_num_id],
                                                             main_coor)
                    if main_tool_movement < mask_dist_metrics:
                        main_tool_coor = (mass_xs[sec_max_num_id], mass_ys[sec_max_num_id])
                        # Update max_num_id solves the main tool switch problem
                        new_max_num_id = sec_max_num_id
                else:
                    # how to set the main tool coor here?
                    # If new main tool shows, we assign it the coor that not the assist tool's coor
                    # Need to change the locate_new_main_tool func
                    # If need to empty main tool coor, do it here
                    track_main_tool = 1
                    assist_tool_coor = (mass_xs[max_num_id], mass_ys[max_num_id])
                    save_no_main_tool_data("Only have the assist tool")
        else:
            main_tool_movement = cal_tools_move_dist(mass_xs[max_num_id], mass_ys[max_num_id], main_coor)
            if main_tool_movement > true_rad * 0.75:
                assist_tool_coor = (mass_xs[max_num_id], mass_ys[max_num_id])
                if len(contour_area_sets) > 1 and sorted_contour_areas[-2] > small_area_metrics / 5:
                    main_tool_move_dist1 = cal_tools_move_dist(mass_xs[sec_max_num_id], mass_ys[sec_max_num_id],
                                                               main_coor)
                    if main_tool_move_dist1 > true_rad * 0.75:
                        track_main_tool = 1
                        save_no_main_tool_data("Can't locate the main tool in multiple masks")
                    else:
                        new_max_num_id = sec_max_num_id
                        main_tool_coor = (mass_xs[new_max_num_id], mass_ys[new_max_num_id])
                else:
                    track_main_tool = 1
                    save_no_main_tool_data("Main tool is out of the view")
            else:
                main_tool_coor = (mass_xs[max_num_id], mass_ys[max_num_id])

        return track_main_tool, assist_tool_coor, main_tool_coor, new_max_num_id

    sorted_contour_areas = sorted(contour_area_sets)
    # When no primary tool in the view, but doesn't consider the assist tool???
    if has_main_tool == 0:
        has_main_tool, main_coor = locate_new_main_tool()
        return has_main_tool, main_coor, assist_coor, max_num_id
    # When the tool is already in the view
    else:
        get_main_tool, assist_coor, main_coor, max_num_id = current_tool_coor()
        return has_main_tool, main_coor, assist_coor, max_num_id


def bad_main_tool_mask_filter(contours_number, max_num_id, frame_num, frame_mask, workbook, row_num):
    def save_no_main_tool_data(text, text_height):
        we.write_excel_table(frame_num, workbook, row_num)
        cv2.putText(frame_mask, text, (20, text_height), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
        cv2.imwrite('C:/D/Clip16SL/clip16' + '_' + str(frame_num) + '.jpg', frame_mask)
        print('No.' + str(frame_num))

    def get_contour_defects():
        interrupt_loop = 0
        hull = cv2.convexHull(contours_number[max_num_id], clockwise=False, returnPoints=False)
        try:
            defects = cv2.convexityDefects(contours_number[max_num_id], hull)
        except cv2.error:
            save_no_main_tool_data("Hull defects error", 140)
            interrupt_loop = 1
            defects = 1
        return interrupt_loop, defects

    def get_ee_convex(contour_defects):
        far_list = []

        for j in range(contour_defects.shape[0]):
            s, e, f, d = contour_defects[j, 0]
            point_dist = np.int(d)
            far_list.append(point_dist)

        new_list = sorted(far_list)
        max_number = far_list.index(new_list[-1])
        s, e, f, d = contour_defects[max_number, 0]
        far = tuple(contours_number[max_num_id][f][0])
        start = tuple(contours_number[max_num_id][s][0])
        end = tuple(contours_number[max_num_id][e][0])
        point_dist = np.int(d)
        cv2.line(frame_mask, start, end, (0, 225, 0), 2)
        cv2.circle(frame_mask, far, 8, (0, 255, 0), 5)
        # use 256 to convert point_dist to dist with pixel unit
        cv2.putText(frame_mask, "Farthest dist: {:.2f}".format(point_dist / 256), (20, 60), cv2.FONT_ITALIC, 0.5,
                    (0, 255, 0))
        return point_dist / 256, far, start, end

    def get_ee_and_far_points(p_far, p_start, p_end):
        k_start = abs((p_far[1] - p_start[1]) / (p_far[0] - p_start[0]))
        k_end = abs((p_far[1] - p_end[1]) / (p_far[0] - p_end[0]))
        if k_start > k_end:
            if k_end / k_start < k_ratio:
                pt1 = p_end
            else:
                pt1 = ((p_end[0] + p_start[0]) / 2, (p_end[1] + p_start[1]) / 2)
        else:
            if k_start / k_end < k_ratio:
                pt1 = p_start
            else:
                pt1 = ((p_end[0] + p_start[0]) / 2, (p_end[1] + p_start[1]) / 2)
        return pt1

    def get_ee_to_body_ratio(point_far, point_1):
        far_to_effector = ((point_far[0] - point_1[0]) ** 2 + (point_far[1] - point_1[1]) ** 2) ** 0.5
        circle_cen = (circle_x, circle_y)
        circle_rad = true_rad
        new_point = GC.circle_line_segment_intersection(circle_cen, circle_rad, point_1, point_far, full_line=True,
                                                        tangent_tol=1e-9)
        if new_point[0][0] > new_point[1][0]:
            new_point = (new_point[0][0], new_point[0][1])
        else:
            new_point = (new_point[1][0], new_point[1][1])
        #cv2.line(frame_mask, point_far, (int(new_point[0]), int(new_point[1])), (0, 0, 255), 2)

        far_to_edge = ((point_far[0] - new_point[0]) ** 2 + (point_far[1] - new_point[1]) ** 2) ** 0.5
        length_ratio = far_to_effector / far_to_edge * 100
        cv2.putText(frame_mask, "effector length ratio: {:.2f}".format(length_ratio), (20, 100), cv2.FONT_ITALIC,
                    0.5, (0, 255, 0))
        return length_ratio

    def pick_good_mask(dist_far_to_ee, point_far, point_start, point_end):
        interrupt_looping = 0
        if dist_far_to_ee > end_effector_detail:
            save_no_main_tool_data("Longer than end_effector_detail, pass", 80)
            interrupt_looping = 1
        else:
            if (dist_far_to_ee) > tool_body_concave:
                cv2.circle(frame_mask, point_start, 8, (255, 255, 0), 5)
                cv2.circle(frame_mask, point_end, 8, (0, 255, 0), 5)
                cv2.putText(frame_mask, "Shorter than end_effector_detail, consider", (20, 80), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
                pt1 = get_ee_and_far_points(point_far, point_start, point_end)
                length_ratio = get_ee_to_body_ratio(point_far, pt1)
                if length_ratio >= length_ratio_metrics:
                    save_no_main_tool_data("Longer than length_ratio_metrics, pass", 120)
                    interrupt_looping = 1
        if interrupt_looping == 0:
            cv2.putText(frame_mask, "Record", (20, 120), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
        return interrupt_looping

    interrupt_loop, defects = get_contour_defects()
    if interrupt_loop:
        return 1
    point_dist, far, start, end = get_ee_convex(defects)
    interrupt_loop = pick_good_mask(point_dist, far, start, end)
    if interrupt_loop:
        return 1
    else:
        return 0





