import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("x/1.jpg")
mean = np.mean(img)
std = np.std(img)
img = img-mean
img = img/std
#提取肺部大致均值
middle = img[100:450,200:450]
mean = np.mean(middle)

# 将图片最大值和最小值替换为肺部大致均值
max = np.max(img)
min = np.min(img)
print(mean,min,max)
img[img==max]=mean
img[img==min]=mean


kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
centers = sorted(kmeans.cluster_centers_.flatten())
threshold = np.mean(centers)
thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
print('kmean centers:',centers)
print ('threshold:',threshold)
'''
kmean centers: [-0.2307924288649088, 1.472218336483015]
threshold: 0.6207129538090531
'''
# # 聚类完成后，清晰可见偏黑色区域为一类，偏灰色区域为另一类。
# image_array = thresh_img
# plt.imshow(image_array,cmap='gray')
# plt.show()

eroded = morphology.erosion(thresh_img)
dilation = morphology.dilation(eroded)
labels = measure.label(dilation)
fig,ax = plt.subplots(2,2,figsize=[8,8])
ax[0,0].imshow(thresh_img,cmap='gray')
ax[0,1].imshow(eroded,cmap='gray')
ax[1,0].imshow(dilation,cmap='gray')
ax[1,1].imshow(labels)  # 标注mask区域切片图
plt.show()
#
# f, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,8))
# ax1.imshow(img,cmap='gray')
# plt.hist(img.flatten(),bins=200)
# plt.show()