import cv2
import numpy


frame = cv2.imread('Clip16_1M/clip16' + '_' + str(1229) + 'M.jpg')
frame1 = cv2.imread('Clip16/clip16' + '_' + str(1229) + '.jpg')
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
rightToolTot = 0
leftToolTot = 0
leftToolTot_Previous = 0
rightToolTot_Previous = 0


for num_of_contours in range(len(contours)):
    # Define Area
    new_contours = []
    points_touch_edge = 0
    # Finish parameters define

    M = cv2.moments(contours[num_of_contours], 0)
    mass_centres_x = int(M['m10'] / M['m00'])
    mass_centres_y = int(M['m01'] / M['m00'])
    area = cv2.contourArea(contours[num_of_contours])

    if mass_centres_x < 507:
        number = len(contours[num_of_contours])
        for i in range(number):
            if contours[num_of_contours][i][0][0] < 507:
                # Blue points show tool boundary
                cv2.circle(frame1, (contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]), radius=1, color=(200, 210, 0), thickness=4)
                dist = (contours[num_of_contours][i][0][0] - 506) ** 2 + (contours[num_of_contours][i][0][1] - 279) ** 2
                if dist >= 410 ** 2 and dist <= 420 ** 2:
                    new_contours.append([[contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]]])
                    points_touch_edge += 1

                # Example 3064, 4759
                elif (contours[num_of_contours][i][0][0]<818 and contours[num_of_contours][i][0][0]>196) and (contours[num_of_contours][i][0][1]<=8 and contours[num_of_contours][i][0][1]>=0):
                    new_contours.append([[contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]]])
                    points_touch_edge += 1

                elif (contours[num_of_contours][i][0][0]<818 and contours[num_of_contours][i][0][0]>196) and (contours[num_of_contours][i][0][1]<=540 and contours[num_of_contours][i][0][1]>=532):
                    new_contours.append([[contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]]])
                    points_touch_edge += 1

        # Try to eliminate random noise
        if points_touch_edge >= 1:
            #Assign Area
            leftTool.append(area)
            # Yellow circle for the mass centre (Recognise as a tool, example 2570)
            cv2.circle(frame1, (mass_centres_x, mass_centres_y), radius=4, color=(0, 255, 255), thickness=4)
            # Green points shown if the tool touch with the lens boundary
            for k in range(len(new_contours)):
                cv2.circle(frame1, (new_contours[k][0][0], new_contours[k][0][1]), radius=1, color=(0, 255, 0), thickness=4)

    elif mass_centres_x > 507:
        number = len(contours[num_of_contours])
        for i in range(number):
            if contours[num_of_contours][i][0][0] > 507:


                cv2.circle(frame1, (contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]), radius=1, color=(200, 210, 0), thickness=4)
                dist = (contours[num_of_contours][i][0][0] - 506) ** 2 + (contours[num_of_contours][i][0][1] - 279) ** 2
                if dist >= 410 ** 2 and dist <= 420 ** 2:
                    new_contours.append([[contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]]])
                    points_touch_edge += 1

                elif (contours[num_of_contours][i][0][0]<818 and contours[num_of_contours][i][0][0]>196) and (contours[num_of_contours][i][0][1]<=8 and contours[num_of_contours][i][0][1]>=0):
                    new_contours.append([[contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]]])
                    points_touch_edge += 1

                elif (contours[num_of_contours][i][0][0]<818 and contours[num_of_contours][i][0][0]>196) and (contours[num_of_contours][i][0][1]<=540 and contours[num_of_contours][i][0][1]>=532):
                    new_contours.append([[contours[num_of_contours][i][0][0], contours[num_of_contours][i][0][1]]])
                    points_touch_edge += 1

        if points_touch_edge >= 1:
            # Assign Area
            rightTool.append(area)
            # Yellow circle for the mass centre (Recognise as a tool)
            cv2.circle(frame1, (mass_centres_x, mass_centres_y), radius=4, color=(0, 255, 255), thickness=4)
            for k in range(len(new_contours)):
                cv2.circle(frame1, (new_contours[k][0][0], new_contours[k][0][1]), radius=1, color=(0, 255, 0), thickness=4)

# Process area change
if len(leftTool):
    leftTool.sort(reverse=True)
    leftToolTot = sum(leftTool[0:2])
    leftTool1 = leftTool[0]
    leftTool2 = leftTool[1]
    if leftToolTot_Previous:
        leftchangerate = (leftToolTot - leftToolTot_Previous) / leftToolTot_Previous * 100
    else:
        # Prevent rightToolTot_Previous = 0
        leftchangerate = 100

cv2.putText(frame1, "Left Tool 1 area: " + str(leftTool1), (20, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
cv2.putText(frame1, "Left Tool 2 area: " + str(leftTool2), (20, 40), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
cv2.putText(frame1, "Left Tool total area: " + str(leftToolTot), (20, 60), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
cv2.putText(frame1, "Left Tool area change rate: {:.2f}%".format(leftchangerate), (20, 80), cv2.FONT_ITALIC, 0.5,
            (0, 255, 0))

if len(rightTool):
    rightTool.sort(reverse=True)
    rightToolTot = sum(rightTool[0:2])
    rightTool1 = rightTool[0]
    rightTool2 = rightTool[1]

rightTool.sort(reverse=True)
rightToolTot = sum(rightTool[0:2])
rightTool1 = rightTool[0]
rightTool2 = rightTool[1]
if rightToolTot_Previous:
    rightchangerate = (rightToolTot - rightToolTot_Previous) / rightToolTot_Previous * 100
else:
    rightchangerate = 100

cv2.putText(frame1, "right Tool 1 area: " + str(rightTool1), (730, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
cv2.putText(frame1, "right Tool 2 area: " + str(rightTool2), (730, 40), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
cv2.putText(frame1, "right Tool total area: " + str(rightToolTot), (730, 60), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
cv2.putText(frame1, "Right Tool area change rate: {:.2f}%".format(rightchangerate), (650, 80), cv2.FONT_ITALIC, 0.5, (0, 255, 0))

cv2.imshow('ImgData/C16-0017-2Tool-ZI.png', frame1)
cv2.waitKey(0)
