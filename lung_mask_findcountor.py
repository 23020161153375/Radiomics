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


image_tmp = cv2.imread('1.jpg',0)
image_ori=image_tmp.copy()
plt.subplot(241), plt.imshow(image_tmp,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]),plt.yticks([])


image_canny = cv2.Canny(image_tmp, 200, 300) # 边缘检测
plt.subplot(242), plt.imshow(image_canny,cmap = 'gray')
plt.title('Edge Image'),plt.xticks([]),plt.yticks([])


# Step2. 二值化
ret, thresh = cv2.threshold(image_tmp, 10, 255, cv2.THRESH_BINARY)
# ret, thresh = cv2.threshold(image_tmp, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素



opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # 闭运
opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)  # 开运算
# Step3. 轮廓提取


plt.subplot(243),plt.imshow(opening)
plt.title('closing-opening'),plt.xticks([]),plt.yticks([])


contour, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Step4. 轮廓绘制
color = cv2.cvtColor(image_tmp, cv2.COLOR_GRAY2BGR)
result = cv2.drawContours(color, contour, -1, (0,255,0), 2) #轮廓用绿色绘制


copyIma=fill_color_demo(result)
plt.subplot(244),plt.imshow(copyIma)
plt.title('floodFill Image'),plt.xticks([]),plt.yticks([])

mask = image_tmp
mask[copyIma[:,:,0]==100]=1
mask[copyIma[:,:,0]!=100]=0
print(image_tmp.shape)
print(mask.shape)

kernel = np.ones((3,3),np.uint8)
kernel2 = np.ones((10,10),np.uint8)
kernel3 = np.ones((20,20),np.uint8)
kernel4 = np.ones((5,5),np.uint8)
kernel5= np.ones((16,16),np.uint8)
erosion = cv2.erode(mask,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel5,iterations =1)

# dilation = cv2.dilate(dilation,kernel4,iterations =1)
# dilation = cv2.dilate(dilation,kernel3,iterations =1)
plt.subplot(245),plt.imshow(mask,cmap = 'gray')
plt.title('mask original'),plt.xticks([]),plt.yticks([])
# plt.subplot(336),plt.imshow(erosion*255)
# plt.title('erosion Image'),plt.xticks([]),plt.yticks([])




# erosion = cv2.erode(dilation,kernel2,iterations = 1)
# dilation = cv2.dilate(erosion,kernel2,iterations = 1)
# image_mask = np.array(dilation)
# image_ori = np.array(image_ori)
# img1_bg = cv2.bitwise_and(image_ori,dilation)
# masked = cv2.bitwise_and(image_tmp, mask)
# mask.reshape(image_tmp.shape)
# roi = cv2.bitwise_and(image_tmp, mask)
plt.subplot(246),plt.imshow(dilation,cmap = 'gray')
plt.title('ROI mask'),plt.xticks([]),plt.yticks([])
# plt.subplot(338),plt.imshow(image_ori,cmap = 'gray')
# plt.title('ori image'),plt.xticks([]),plt.yticks([])
plt.subplot(247),plt.imshow(image_ori*dilation,cmap = 'gray')
plt.title('ROI image'),plt.xticks([]),plt.yticks([])

plt.savefig("result.jpg")
plt.show()
