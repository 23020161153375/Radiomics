import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import cv2
import dicom
import matplotlib.pyplot as plt

def lung_mask(image_path):
    # working_path = "/opt/code/medical/lung_preprocessing/slice_and_mask/"
    # file_list=glob(working_path+"images_*.npy")
    #以1.3.6.1.4.1.14519.5.2.1.6279.6001.154677396354641150280013275227.mhd肺结节为例
    # img_file ="4.npy"
    print(image_path)
    '''
    '/opt/code/medical/lung_preprocessing/slice_and_mask/images_0024_0199.npy'
    '''
    # f = dicom.read_file(img_file)
    imgs_to_process = np.load(image_path).astype(np.float64)
    # imgs_to_process = dicom.read_file(img_file)
    # print(imgs_to_process.shape) # 病历离肺结节最近的三个切片
    '''
    (3, 512, 512)
    '''

    # img = cv2.imread("x/1.jpg")
    img = imgs_to_process[0] # 以一张切片为例
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    #提取肺部大致均值
    middle = img[100:400,200:400]
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

    # # # 聚类完成后，清晰可见偏黑色区域为一类，偏灰色区域为另一类。
    # image_array = thresh_img
    # plt.imshow(image_array,cmap='gray')
    # plt.show()

    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels) # 获取连通区域
    #
    # 设置经验值，获取肺部标签
    good_labels = []
    for prop in regions:
        B = prop.bbox
        print(B)
        if B[2]-B[0]<400 and B[3]-B[1]<400 and B[0]>40 and B[2]<400:
            good_labels.append(prop.label)
    # '''
    # (0L, 0L, 512L, 512L)
    # (190L, 253L, 409L, 384L)
    # (200L, 110L, 404L, 235L)
    # '''
    # # 根据肺部标签获取肺部mask，并再次进行’膨胀‘操作，以填满并扩张肺部区域
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    eroded = morphology.erosion(mask,np.ones([4,4]))
    mask = morphology.dilation(eroded,np.ones([10,10])) # one last dilation
    new_size = [512,512]   # we're scaling back up to the original size of the image
    output = img*mask
    # fig,ax = plt.subplots(2,2,figsize=[10,10])
    # ax[0,0].imshow(img)  # CT切片图
    # ax[0,1].imshow(img,cmap='gray')  # CT切片灰度图
    # ax[1,0].imshow(mask,cmap='gray')  # 标注mask，标注区域为1，其他为0
    # ax[1,1].imshow(img*mask,cmap='gray')  # 标注mask区域切片图
    # plt.show()
    return output

if __name__=='__main__':
    img_path = '1.npy'
    output= lung_mask(img_path)
    np.save("result.npy",output)
    result = np.load("result.npy")

    plt.imshow(result)
    plt.show()
