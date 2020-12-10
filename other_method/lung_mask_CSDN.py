# -*- coding:utf-8 -*-
'''
this script is used for basic process of lung 2017 in Data Science Bowl
'''

import glob
import os
import pandas as pd
import SimpleITK as sitk

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np

def get_segmented_lungs(im, plot=True):

    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(2, 4, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.
    '''
    binary = im < 604
    if plot == True:
        plots[0,0].axis('off')
        plots[0,0].set_title('binary image')
        plots[0,0].imshow(binary, cmap=plt.cm.bone)

    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[0,1].axis('off')
        plots[0,1].set_title('after clear border')
        plots[0,1].imshow(cleared, cmap=plt.cm.bone)

    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[0,2].axis('off')
        plots[0,2].set_title('found all connective graph')
        plots[0,2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[0,3].axis('off')
        plots[0,3].set_title(' Keep the labels with 2 largest areas')
        plots[0,3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[1,0].axis('off')
        plots[1,0].set_title('seperate the lung nodules attached to the blood vessels')
        plots[1,0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[1,1].axis('off')
        plots[1,1].set_title('keep nodules attached to the lung wall')
        plots[1,1].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[1,2].axis('off')
        plots[1,2].set_title('Fill in the small holes inside the binary mask of lungs')
        plots[1,2].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[1,3].axis('off')
        plots[1,3].set_title('Superimpose the binary mask on the input image')
        plots[1,3].imshow(im, cmap=plt.cm.bone)
    plt.show()
    return im

if __name__=='__main__':
    img = np.load('3.npy')
    res = get_segmented_lungs(img[0],plot=True)
