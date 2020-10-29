import cv2

leftToolTot_Previous = 0
rightToolTot_Previous = 0
Area_notchange = 0

for frames in range(5901, 5908, 1):
    frame = cv2.imread('Clip16_1M/clip16' + '_' + str(frames) + 'M.jpg')
    frame1 = cv2.imread('Clip16_1D/clip16' + '_' + str(frames) + 'D.jpg')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the boundary for edge-contacted tool detection
    center_coordinatesleft = (506, 279)
    radius2 = 418
    radius3 = 410
    color1 = (0, 0, 255)
    cv2.circle(frame1, center_coordinatesleft, radius2, color1, thickness=1)
    cv2.circle(frame1, center_coordinatesleft, radius3, color1, thickness=1)

    pt1 = (196, 0)
    pt2 = (818, 8)
    cv2.rectangle(frame1, pt1, pt2, color=(0, 0, 255), thickness=1)

    pt3 = (196, 532)
    pt4 = (818, 540)
    cv2.rectangle(frame1, pt3, pt4, color=(0, 0, 255), thickness=1)

    x1, y1 = 507, 0
    x2, y2 = 507, 540
    cv2.line(frame1, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
    # Finish Drawing

    rightTool1 = 0
    rightTool2 = 0
    rightTool = [0, 0]
    leftTool1 = 0
    leftTool2 = 0
    leftTool = [0, 0]
    twoConnectedTool_area = [0]
    rightToolTot = 0
    leftToolTot = 0

    number_of_contours = len(contours)
    for num_of_contours in range(len(contours)):
        # Define Area
        new_contours = []
        points_touch_edge = 0
        points_touch_another_edge = 0
        # Finish parameters define

        M = cv2.moments(contours[num_of_contours], 0)
        if M['m00']:
            mass_centres_x = int(M['m10'] / M['m00'])
            mass_centres_y = int(M['m01'] / M['m00'])
        else:
            continue
        area = cv2.contourArea(contours[num_of_contours])

        if mass_centres_x <= 507:
            number = len(contours[num_of_contours])
            for i in range(number):
                # Detect whether two tools connect together
                if contours[num_of_contours][i][0][0] > 818:
                    dist = (contours[num_of_contours][i][0][0] - 506) ** 2 + (
                            contours[num_of_contours][i][0][1] - 279) ** 2
                    if dist >= 410 ** 2 and dist <= 430 ** 2:
                        points_touch_another_edge += 1
                        cv2.circle(frame1, (contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]),
                                   radius=1, color=(130, 100, 0), thickness=4)

                else:
                    if contours[num_of_contours][i][0][0] < 196:
                        # Blue points show tool boundary
                        cv2.circle(frame1, (contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]),
                                   radius=1, color=(200, 210, 0), thickness=4)
                        dist = (contours[num_of_contours][i][0][0] - 506) ** 2 + (
                                    contours[num_of_contours][i][0][1] - 279) ** 2
                        # Detect left and right boundaries
                        if dist >= 410 ** 2 and dist <= 430 ** 2:
                            new_contours.append([[contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]]])
                            points_touch_edge += 1

                    if contours[num_of_contours][i][0][0] < 507:
                        # Detect up boundary, Example 3064, 4759
                        if (contours[num_of_contours][i][0][0] < 818 and contours[num_of_contours][i][0][0] > 196) and (
                                contours[num_of_contours][i][0][1] <= 8 and contours[num_of_contours][i][0][1] >= 0):
                            new_contours.append([[contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]]])
                            points_touch_edge += 1

                        # Detect bottom boundary
                        elif (contours[num_of_contours][i][0][0] < 818 and contours[num_of_contours][i][0][0] > 196) and (
                                contours[num_of_contours][i][0][1] <= 540 and contours[num_of_contours][i][0][1] >= 532):
                            new_contours.append([[contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]]])
                            points_touch_edge += 1


            # Try to eliminate random noise, only keep masks inserted from the edge
            if points_touch_another_edge >= 3:
                twoConnectedTool_area.append(area)
                cv2.circle(frame1, (mass_centres_x, mass_centres_y), radius=4, color=(0, 0, 255), thickness=4)

            elif points_touch_edge >= 5:
                # Assign Area
                leftTool.append(area)
                # Yellow circle for the mass centre (Recognise as a tool, example 2570)
                cv2.circle(frame1, (mass_centres_x, mass_centres_y), radius=4, color=(0, 255, 255), thickness=4)
                # Green points shown if the tool touch with the lens boundary
                for k in range(len(new_contours)):
                    cv2.circle(frame1, (new_contours[k][0][0], new_contours[k][0][1]), radius=1, color=(0, 255, 0),
                               thickness=4)


        elif mass_centres_x > 507:
            number = len(contours[num_of_contours])
            for i in range(number):
                if contours[num_of_contours][i][0][0] < 196:
                    dist = (contours[num_of_contours][i][0][0] - 506) ** 2 + (
                            contours[num_of_contours][i][0][1] - 279) ** 2
                    if dist >= 410 ** 2 and dist <= 430 ** 2:
                        points_touch_another_edge += 1
                        cv2.circle(frame1, (contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]),
                                   radius=1, color=(130, 100, 0), thickness=4)

                else:
                    if contours[num_of_contours][i][0][0] > 818:
                        cv2.circle(frame1, (contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]),
                                   radius=1, color=(200, 210, 0), thickness=4)
                        dist = (contours[num_of_contours][i][0][0] - 506) ** 2 + (
                                    contours[num_of_contours][i][0][1] - 279) ** 2
                        if dist >= 410 ** 2 and dist <= 430 ** 2:
                            new_contours.append([[contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]]])
                            points_touch_edge += 1

                    if contours[num_of_contours][i][0][0] > 507:
                        if (contours[num_of_contours][i][0][0] < 818 and contours[num_of_contours][i][0][0] > 196) and (
                                contours[num_of_contours][i][0][1] <= 8 and contours[num_of_contours][i][0][1] >= 0):
                            new_contours.append([[contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]]])
                            points_touch_edge += 1

                        elif (contours[num_of_contours][i][0][0] < 818 and contours[num_of_contours][i][0][0] > 196) and (
                                contours[num_of_contours][i][0][1] <= 540 and contours[num_of_contours][i][0][1] >= 532):
                            new_contours.append([[contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]]])
                            points_touch_edge += 1

            if points_touch_another_edge >= 3:
                twoConnectedTool_area.append(area)
                cv2.circle(frame1, (mass_centres_x, mass_centres_y), radius=4, color=(0, 0, 255), thickness=4)

            elif points_touch_edge >= 5:
                # Assign Area
                rightTool.append(area)
                # Yellow circle for the mass centre (Recognise as a tool)
                cv2.circle(frame1, (mass_centres_x, mass_centres_y), radius=4, color=(0, 255, 255), thickness=4)
                for k in range(len(new_contours)):
                    cv2.circle(frame1, (new_contours[k][0][0], new_contours[k][0][1]), radius=1, color=(0, 255, 0),
                               thickness=4)

    twoConnectedTool_area.sort(reverse=True)
    if twoConnectedTool_area[0] > 0:
        cv2.putText(frame1, "Two tools touch each other, area: " + str(twoConnectedTool_area[0]), (20, 100), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
        # When two tools touch each other, the zoom ratio should be frozen

    # From all contours in one frame, if find tools on the left side
    if len(leftTool) > 2:
        leftTool.sort(reverse=True)
        leftToolTot = sum(leftTool[0:2])
        leftTool1 = leftTool[0]
        leftTool2 = leftTool[1]
        # If the tool is already in the view
        if leftToolTot_Previous:
            leftToolChange = leftToolTot - leftToolTot_Previous
            leftchangerate = leftToolChange / leftToolTot_Previous * 100

            # Prevent random noise, I assume if the change rate is too high, then there should be some noise. The value is set from image 183-185
            if abs(leftchangerate) >= 1500:
                # Just for test, the value given to the current total value should consider the moving trend(Using Kalman filter)
                leftToolTot = leftToolTot_Previous
                # The change rate should keep the previous value if lost tracking?
                leftchangerate = 0
        # If the tool is newly inserted, 200% is signal for new tool insert?
        else:
            leftchangerate = 200

    # If no tool
    else:
        # Prevent if the detect mask contour doesn't touch the lens edge (image840-873)
        if leftToolTot_Previous > 12000:
            leftToolTot = leftToolTot_Previous
            # The change rate should keep the previous value if lost tracking?
            leftchangerate = 0
        elif leftToolTot_Previous > 1000 and leftToolTot_Previous < 10000:
            leftchangerate = -200
            leftToolTot = 0
        else:
            leftchangerate = 0
            leftToolTot = 0
        leftTool1 = 0
        leftTool2 = 0

    # From all contours in one frame, if find tools on the right side
    if len(rightTool) > 2:
        rightTool.sort(reverse=True)
        rightToolTot = sum(rightTool[0:2])
        rightTool1 = rightTool[0]
        rightTool2 = rightTool[1]

        if rightToolTot_Previous:
            rightToolChange = rightToolTot - rightToolTot_Previous
            rightchangerate = rightToolChange / rightToolTot_Previous * 100

            if -rightToolChange > 12000 and rightToolTot == 0:
                rightToolTot = rightToolTot_Previous
                rightchangerate = 0

            if abs(rightchangerate) >= 1500:
                rightToolTot = rightToolTot_Previous
                rightchangerate = 0
        else:
            rightchangerate = 200

    else:
        if rightToolTot_Previous > 12000:
            rightToolTot = rightToolTot_Previous
            rightchangerate = 0
        elif rightToolTot_Previous > 1000 and rightToolTot_Previous < 10000:
            rightchangerate = -200
            rightToolTot = 0
        else:
            rightchangerate = 0
            rightToolTot = 0
        rightTool1 = 0
        rightTool2 = 0


    # If the area is frozen for 5 frames, force it to be zero
    if rightToolTot_Previous > 0 and rightToolTot_Previous - rightToolTot == 0:
        Area_notchange += 1
        if Area_notchange > 4:
            rightToolTot = 0
            Area_notchange = 0

    leftToolTot_Previous = leftToolTot
    rightToolTot_Previous = rightToolTot

    cv2.putText(frame1, "Left Tool 1 area: " + str(leftTool1), (20, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    cv2.putText(frame1, "Left Tool 2 area: " + str(leftTool2), (20, 40), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    cv2.putText(frame1, "Left Tool total area: " + str(leftToolTot), (20, 60), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    cv2.putText(frame1, "Left Tool area change rate: {:.2f}%".format(leftchangerate), (20, 80), cv2.FONT_ITALIC, 0.5,
                (0, 255, 0))

    cv2.putText(frame1, "right Tool 1 area: " + str(rightTool1), (700, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    cv2.putText(frame1, "right Tool 2 area: " + str(rightTool2), (700, 40), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    cv2.putText(frame1, "right Tool total area: " + str(rightToolTot), (700, 60), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    cv2.putText(frame1, "Right Tool area change rate: {:.2f}%".format(rightchangerate), (650, 80), cv2.FONT_ITALIC, 0.5,
                (0, 255, 0))

    cv2.imwrite('Clip16_GUITest/clip16' + '_' + str(frames) + '.jpg', frame1)
    cv2.waitKey(0)