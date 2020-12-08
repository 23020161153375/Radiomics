# -*- coding: cp936 -*-
import cv2
from matplotlib import pyplot as plt
import numpy as np
# Step1. 读入图像


def fill_color_demo(image):
    copyIma = image.copy()
    h, w = image.shape[:2]
    print(h, w)
    mask = np.zeros([h+2, w+2], np.uint8)
    cv2.floodFill(copyIma, mask, (150,300), (100, 100, 100), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    return copyIma

def lung_mask(image):
    image_tmp = image.copy()
    image_ori= image.copy()

    # Step2. 二值化
    ret, thresh = cv2.threshold(image_tmp, 10, 255, cv2.THRESH_BINARY)
    # ret, thresh = cv2.threshold(image_tmp, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素

    opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # 闭运
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)  # 开运算
    # Step3. 轮廓提取

    contour, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Step4. 轮廓绘制
    color = cv2.cvtColor(image_tmp, cv2.COLOR_GRAY2BGR)
    result = cv2.drawContours(color, contour, -1, (0, 255, 0), 2)  # 轮廓用绿色绘制

    copyIma = fill_color_demo(result)

    mask = image_tmp
    mask[copyIma[:, :, 0] == 100] = 1
    mask[copyIma[:, :, 0] != 100] = 0

    kernel = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((10, 10), np.uint8)
    kernel3 = np.ones((20, 20), np.uint8)
    kernel4 = np.ones((5, 5), np.uint8)
    kernel5 = np.ones((16, 16), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel5, iterations=1)
    output_image = image_ori * dilation
    return output_image



if __name__=='__main__':
    image_tmp = cv2.imread('1.jpg',0)
    result = lung_mask(image_tmp)
    cv2.imwrite("result_one.jpg",result)
