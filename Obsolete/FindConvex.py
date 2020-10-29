import cv2
import numpy as np
from math import atan

img = cv2.imread('E:/Clip31-40/Clip33_1M/clip' + str(33) + '_' + str(618) + 'M.jpg')
#img = cv2.imread('000.PNG', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
cv2.drawContours(img, cnt, -1, (0, 0, 255), 3)
hull = cv2.convexHull(cnt, clockwise=False, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)

#print(hull)

#cv2.drawContours(img, cnt, -1, (0, 0, 255), 3)
far_list = []

#'''
for j in range(defects.shape[0]):
    s, e, f, d = defects[j, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    point_dist = np.int(d)
    far_list.append(point_dist)
    # 用红色连接凸缺陷的起始点和终止点
    cv2.line(img, start, end, (0, 0, 225), 2)
    # 用蓝色最远点画一个圆圈
    #cv2.circle(img, far, 5, (225, 0, 0), -1)

new_list = sorted(far_list)
max_number = far_list.index(new_list[-1])
s, e, f, d = defects[max_number, 0]
far = tuple(cnt[f][0])
start = tuple(cnt[s][0])
end = tuple(cnt[e][0])
point_dist = np.int(d)
cv2.line(img, start, end, (0, 225, 0), 2)
cv2.circle(img, far, 8, (0, 255, 0), 5)

k1 = (start[1] - end[1]) / (start[0] - end[0])
a1 = atan(k1)
b1 = b2 = -1
c1 = start[1] - k1 * start[0]

pvx = (b1 * b1 * far[0] - k1 * b1 * far[1] - k1 * c1) / (k1 * k1 + b1 * b1)
pvy = (-k1 * b1 * far[0] + k1 * k1 * far[1] - b1 * c1) / (k1 * k1 + b1 * b1)

print(((far[0]-pvx)**2+(far[1]-pvy)**2)**0.5)
print(point_dist/256)

cv2.line(img, far, (int(pvx), int(pvy)), (0, 225, 0), 2)


#print(new_list)

#print(start)

#print(hull)
#length = len(hull)
#for i in range(len(hull)):
    #print(hull[i][0])
    #print(hull[(i+1)%length][0])
    #print(' ')
    #cv2.line(img, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,255,0), 2)
#'''
cv2.imshow('line', img)
cv2.waitKey()
