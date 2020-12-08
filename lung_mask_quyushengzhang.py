import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import copy as cp
import random
import math
import cv2

def find_points(img,num):
    shape = img.shape
    center_point_left = [int(0.25 * shape[0]), int(0.75 * shape[1])]
    center_point_right = [int(0.75 * shape[0]), int(0.75 * shape[1])]
    left_point_list = []
    right_point_list = []

    while len(left_point_list) < num:
        r_x = random.randint(-100, 100)
        left_point = [max(0, center_point_left[0] + r_x), min(shape[1], center_point_left[1] + r_x)]

        if th1[left_point[0], left_point[1]] == 0:
            left_point_list.append(left_point)

    while len(right_point_list) < num:
        r_x = random.randint(-100, 100)

        right_point = [min(shape[0], center_point_right[0] + r_x), min(shape[1], center_point_right[1] + r_x)]

        if img[right_point[0], right_point[1]] == 0:
            right_point_list.append(right_point)
    print("left_point_list=",left_point_list)
    print("right_point_list=",right_point_list)
    return left_point_list,right_point_list

img = cv2.imread("1.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, th1 = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
left_point_list,right_point_list = find_points(th1,3)




cv2.waitKey(0)
# # print(zeros.shape)
# print(ret)
# cv2.imshow("th1",th1)
# cv2.waitKey()