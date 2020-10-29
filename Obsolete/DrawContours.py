import cv2

#frame = cv2.imread('Clip24_1M\clip24_8M.jpg')
frame1 = cv2.imread('E:/Clip23/clip23_2000.jpg')
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
#contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

x1, y1 = 290+15, 0
x2, y2 = 290+15, 540
cv2.line(frame1, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)

x3, y3 = 716-15, 0
x4, y4 = 716-15, 540
cv2.line(frame1, (x3, y3), (x4, y4), (0, 0, 255), thickness=1)

x5, y5 = 600, -10
x6, y6 = 500, 30
cv2.line(frame1, (x5, y5), (x6, y6), (0, 0, 255), thickness=1)

cv2.rectangle(frame1, (196+15, 0), (818-15, 10), (0, 0, 255), thickness=1)
cv2.rectangle(frame1, (196+15, 530), (818-15, 540), (0, 0, 255), thickness=1)

circle_center = (503, 252)
two_radiuses = [333, 357]
true_rad = 347

cv2.circle(frame1, circle_center, two_radiuses[0], (255, 0, 0), thickness=1)
cv2.circle(frame1, circle_center, two_radiuses[1], (255, 0, 0), thickness=1)
cv2.circle(frame1, circle_center, true_rad, (10, 100, 250), thickness=1)

'''
for num_of_contours in range(len(contours)):
    M = cv2.moments(contours[num_of_contours], 0)
    if M['m00']:
        mass_centres_x = int(M['m10'] / M['m00'])
        mass_centres_y = int(M['m01'] / M['m00'])

    cv2.circle(frame1, (mass_centres_x, mass_centres_y), radius=4, color=(0, 255, 255), thickness=4)
'''
cv2.imshow('A', frame1)
cv2.waitKey(0)