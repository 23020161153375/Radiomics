import numpy as np
from skimage import morphology
import cv2
import matplotlib.pyplot as plt
import body_mask

def del_small_area_from_mask(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=None)
    del_label_list = []
    for i in range(num_labels):
        if stats[i][4] < 2000:
            del_label_list.append(i)
    mask_Connected = np.ones(mask.shape, dtype='uint8')
    for i in del_label_list:
        mask_Connected[labels == i] = 0

    output_mask = mask_Connected * mask
    return output_mask

def lung_mask_HU(image,plot=True):
    fig, ax = plt.subplots(2, 3, figsize=[10, 10])

    image= image[0]

    mask = np.zeros(image.shape,dtype= 'uint8')
    #Because the HU of lung is around -270~ -830
    mask[image >-270 ]=1
    # mask[image >-200] = 0
    mask[image < -900] = 1
    # mask[image < -900] = 0

    mask = morphology.erosion(mask, np.ones([4, 4]))
    mask = morphology.dilation(mask, np.ones([8, 8]))  # one last dilation
    deled_mask = del_small_area_from_mask(mask)



    mask_convert = cv2.bitwise_not(deled_mask)-254
    mask_convert = morphology.dilation(mask_convert, np.ones([8, 8]))  # one last dilation
    mask_final = del_small_area_from_mask(mask_convert)

    output = image * mask_final
    if plot == True:
        ax[0, 0].set_title('Original')
        ax[0, 0].imshow(image)
        ax[0, 1].set_title('First mask')
        ax[0, 1].imshow(mask)
        ax[0, 2].set_title('Del small part')
        ax[0, 2].imshow(deled_mask)
        ax[1, 0].set_title('Mask convert')
        ax[1, 0].imshow(mask_convert)
        ax[1, 1].set_title('Mask final')
        ax[1, 1].imshow(mask_final)
        ax[1, 2].set_title('Output')
        ax[1, 2].imshow(output)
        plt.show()
    return output
if __name__=='__main__':
    array = np.load('../3.npy')
    body = body_mask.apply_body_mask_and_bound(array,apply_bound=False)
    print("ROI image shape",body.shape)
    output= lung_mask_HU(body)
    np.save("result.npy",output)
    result = np.load("result.npy")
    plt.imshow(result)
    plt.show()
