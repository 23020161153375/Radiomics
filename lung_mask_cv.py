"""
完成分水岭算法步骤：
1、加载原始图像
2、阈值分割，将图像分割为黑白两个部分
3、对图像进行开运算，即先腐蚀在膨胀
4、对开运算的结果再进行 膨胀，得到大部分是背景的区域
5、通过距离变换 Distance Transform 获取前景区域
6、背景区域sure_bg 和前景区域sure_fg相减，得到即有前景又有背景的重合区域
7、连通区域处理
8、最后使用分水岭算法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step1: 加载图像
img = cv2.imread('1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# plt.imshow(gray)
# plt.show()

# Step2: 阈值分割，将图像分为黑白两部分
ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# Step3: 开运算-先腐蚀再膨胀
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Step4: 对“开运算”的结果进行膨胀，得到大部分都是背景的区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Step5: 通过distanceTransform获取前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

# Step6: sure_bg与sure_fg相减,得到既有前景又有背景的重合区域
sure_fg = np.uint8(sure_fg)
# 此区域和轮廓区域的关系未知
unknown = cv2.subtract(sure_bg, sure_fg)

# Step7: 连通区域处理
# 对连通区域进行标号  序号为 0- N-1
ret, markers = cv2.connectedComponents(sure_fg, connectivity=8)
# OpenCV 分水岭算法对物体做的标注必须都 大于1，背景为标号为0,因此对所有markers加1变成了1-N
markers = markers + 1

# 去掉属于背景区域的部分（即让其变为0，成为背景）
# 此语句的Python语法 类似于if ，“unknow==255” 返回的是图像矩阵的真值表。
# 现在整个markers里面有：0，1，2，3 ..... N
# 0：代表unknown, 1,2,3,...代表就是连通区域

markers[unknown == 255] = 0

# Step8: 分水岭算法
markers = cv2.watershed(img, markers)  # 分水岭算法后，所有轮廓的像素点被标注为  -1
# 现在整个markers里面有：-1, 0, 1
# -1:代表分水岭
lung_mask = img
img[markers == -1] = [0, 0, 255]  # 标注为-1 的像素点标红

lung_mask[markers==1]=[0,0,0]
# ret, thresh_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(lung_mask)
img1_bg = cv2.bitwise_and(img, img, mask=mask_inv)
# ret, thresh_mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
# 画图
plt.subplot(2, 3, 1), plt.imshow(sure_bg, 'gray'), plt.title('background'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(sure_fg, 'gray'), plt.title('frontground'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.imshow(unknown, 'gray'), plt.title('unknown'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4), plt.imshow(markers, 'gray'), plt.title('makers'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5), plt.imshow(img), plt.title('img'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 6), plt.imshow(img1_bg), plt.title('img_cut'), plt.xticks([]), plt.yticks([])
plt.show()