import SimpleITK as sitk 
import numpy as np 

import os
import matplotlib.pyplot as plt 

def find_images_data(case_path):
	''' Find a series of slices data 
	
	Return a list of tuples which consists of numpy ndarray and its imageSOP_UID 
	'''
	images_array_uid = []
	for filename in os.listdir(case_path):
		fp = os.path.join(case_path,filename)
		if os.path.isfile(fp) and ".dcm" in fp:
			images_array_uid.append(find_image_data(fp))
	if len(images_array_uid) > 0:
		return images_array_uid
	else:		 
		raise FileNotFoundError("THERE IS NO DICOM FILES IN case_path:\n"+case_path)

def find_image_data(slice_path):
	'''Directly read a single DICOM file.

	The case_path should be a full path of DICOM file,and it will return the Numpy ndaary 
	of this image and any other DICOM information in its header. (if desired)
	'''
	img = sitk.ReadImage(slice_path)
	#print_image_info(img)

	#extract imageSOP_UID tag,or formally SOP Instance UID (0008,0018) 
	imageSOP_UID_key = "0008|0018"
	imageSOP_UID = img.GetMetaData(imageSOP_UID_key)
	#print("imageSOP_UID:",imageSOP_UID)

	#(1,512,512)
	#print(sitk.GetArrayFromImage(img).shape)

	return sitk.GetArrayFromImage(img),imageSOP_UID

def find_xml_file(case_path):
	'''Find the (only) XML file in the given directory
	'''

	for filename in os.listdir(case_path):
		fp = os.path.join(case_path,filename)
		if os.path.isfile(fp) and ".xml" in filename:
			return fp
	raise FileNotFoundError("No XML FILE WAS FOUND IN CASE PATH: \n" + case_path)

def print_image_info(image):
	#print("Number of images in total:",len(image))
	print("Image Sequences Size:",image.GetSize())
	#print(image.GetSize())

	print("Image Size",image.GetSize())
	print("W:",image.GetWidth())
	print("H:",image.GetHeight())
	print("D:",image.GetDepth())

	print("Meta-data(DICOM INFO)")
	for key in image.GetMetaDataKeys():
	        print("\"{0}\":\"{1}\"".format(key, image.GetMetaData(key)))

def show_image(image_array_path):
	'''An aux function to show images that have been already cut
	'''
	
	for filename in os.listdir(image_array_path):
		fp = os.path.join(image_array_path,filename)
		if os.path.isfile(fp) and ".npy" in filename:
			image_array = np.load(fp)
			img = sitk.GetImageFromArray(image_array,isVector = True)
			
			sitk.Show(img,"sample image")


if __name__ == '__main__':
	#test
	#
	# IComs_path = "D:\cancer\gan\lung\LIDC_dataset\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-30178\3000566.000000-03192"

	# case_path = r"D:\cancer\gan\lung\LIDC_dataset\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-30178\3000566.000000-03192"
	# # image_array = find_images_data(case_path)
	# print(image_array)

	'''
	We accasionally found that when a module is imported 
	it will be executed at first in runtime 
	'''
	#
	# # #
	image_array_path = r"D:/cancer/gan/lung/roi/1.3.6.1.4.1.14519.5.2.1.6279.6001.129007566048223160327836686225/"
	show_image(image_array_path)


	# img = find_image_data("Nodule_001.npy")
	# image_array = np.load("Nodule_001.npy")
	# img = sitk.GetImageFromArray(image_array,isVector = True)
	# print_image_info(img)
	# print(img)

	'''
	slice_path = r"F:/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192\000001.dcm"
	find_image_data(slice_path)
	'''