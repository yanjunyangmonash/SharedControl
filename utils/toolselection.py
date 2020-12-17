import constant
from math import atan2, asin, pi
import cv2
import numpy as np
from utils import write_excel as we

# Laparoscopic view geo parameters
video_num = 50
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


def tool_mask_selection(contours_number):
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
    we.write_excel_table(frame_num, workbook, row_num, pre_width, pre_length, pre_LW_Ratio)
    cv2.putText(frame_mask, "Two tools contact together", (20, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    cv2.imwrite('C:/D/Clip16SL/clip16' + '_' + str(frame_num) + '.jpg', frame_mask)
    print('No.' + str(frame_num))


def left_tool_filter(row_num, frame_num, workbook, frame_mask):
    # Filter out auxiliary tool mask
    we.write_excel_table(frame_num, workbook, row_num, 0, 0, 0)
    cv2.putText(frame_mask, "No Tools on the right side", (20, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
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
    sec_max_num_id = 0
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
