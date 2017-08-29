from xml.dom.minidom import parse
import xml.dom.minidom
import os
import sys
import numpy as np
import random

import files_reader


def gen_roi_images(case_path,output_path):
	'''Generate a series of roi images saved as npy files 

	Bisides,it generates propertional number of plain images 
	wherever a roi image is generated in a certain slice.
	'''
	fp = files_reader.find_xml_file(case_path)
	images_data = files_reader.find_images_data(case_path)

	DOMTree = parse(fp)

	libcReadMessage = DOMTree.documentElement
	SeriesInstanceUid = libcReadMessage.getElementsByTagName("SeriesInstanceUid")[0].firstChild.data

	#Get the first radiogist 's contours(and assume that the four radiogists ' contours on the 
	#unbind condition are somewhat the same).
	readingSession \
		= libcReadMessage.getElementsByTagName("readingSession")[0]

	unblindedReadNodules \
		= readingSession.getElementsByTagName("unblindedReadNodule")


	for nodule in unblindedReadNodules:

		#All "ID-tag" should be tramslated into strings immediately 
		noduleID = nodule.getElementsByTagName("noduleID")[0].firstChild.data
		print(noduleID)
		rois = nodule.getElementsByTagName("roi")
		#search_start = 0
		for roi in rois:

			#If there is only a signle roi in rois, then the size of nodule on which we currently focus 
			#is less than 3mm. Actually you can seperate these small nodules from the bigger with size > 3mm.
			#(From latter experiments we conclude that it may be not a good idea to merge these two kinds of noudles into one category)

			imageSOP_UID = roi.getElementsByTagName("imageSOP_UID")[0].firstChild.data
			inclusion = roi.getElementsByTagName("inclusion")[0]
			if inclusion.firstChild.data == "TRUE":
				#This is ultimate outer boundary of a noudle
				sliceIndex = search_image_by_UID(images_data, imageSOP_UID)
				print("SliceIndex:",sliceIndex)

				#Next time the search will start from the adjacent slice (in a noudule )
				#search_start = sliceIndex + 1

				image_array = images_data[sliceIndex][0]
				roi_image_array = roi_region_cut(roi, image_array)

				#Save roi image array
				roi_image_dir = os.path.join(output_path,"roi",SeriesInstanceUid)
				if not os.path.exists(roi_image_dir):
					#os.mkdir(roi_image_dir)
					os.makedirs(roi_image_dir)

				#File name: roi.#readerID#.#nouduleID#.#sliceIndex#.npy
				np.save(os.path.join(roi_image_dir,"0."+noduleID+"."+str(sliceIndex)+".npy"),roi_image_array)

				#Simultaneously find and save non-noudule images 
				plain_image_dir = os.path.join(output_path,"plain",SeriesInstanceUid)
				if not os.path.exists(plain_image_dir):
					#os.mkdir(plain_image_dir)
					os.makedirs(plain_image_dir)

				plain_image_path = os.path.join(plain_image_dir,"0."+noduleID+"."+str(sliceIndex)+".npy")

				gen_plain_images(image_array, plain_image_path,2)

def gen_plain_images(image_array,full_output_path,factor = 1):
	'''Generate plain image.

	This will be called in gen_roi_images function for the sake of a 
	more concise piece of code. Therefore it is not supported to call 
	this function indepedantly amd directly.
	'''
	if not full_output_path.endswith(".npy"):
		raise ValueError("It is expected to saved images as Numpy npy files.\n"+ full_output_path)

	for i in range(1,factor + 1):
		#upper bound of x (inclusive)
		xBound = image_array.shape[2] - DNN_INPUT_SIDE_LEN
		xMin = random.randint(0,xBound)

		#upper bound of y (inclusive)
		yBound = image_array.shape[1] - DNN_INPUT_SIDE_LEN
		yMin = random.randint(0,yBound)

		print("GENERATE PLAINN-IMAGE: Y1={}, Y2={}, X1={}, X2={}". \
			format(yMin,yMin + DNN_INPUT_SIDE_LEN,xMin,xMin + DNN_INPUT_SIDE_LEN))

		target_image_array = \
			image_array[0, yMin : yMin + DNN_INPUT_SIDE_LEN, xMin : xMin + DNN_INPUT_SIDE_LEN]

		# Insert ".i" before ".npy" tail of full_output_path
		new_output_path = full_output_path[0:len(full_output_path) - 4] + "."+ str(i) + ".npy"
		print("NEW PATH:",new_output_path)
		np.save(new_output_path, target_image_array)

def search_image_by_UID(images_data,imageSOP_UID,start = 0):
	'''Do image matching

	Parameter "start" is the search origin in "images_data"
	array. But latter we found that images data are not in a order by uid,
	in this case ,the "start" parameter makes no difference.
 	'''
	number_of_slices = len(images_data)
	current_index = start % number_of_slices
	current_uid = images_data[current_index][1]

	while current_uid != imageSOP_UID and current_index <= start + number_of_slices:
		current_index += 1
		current_uid = images_data[current_index % number_of_slices][1]
	else:
		if current_uid == imageSOP_UID:
			print("It takes " ,current_index - start + 1,"step for searching  a image." )
			return current_index % number_of_slices
		else:
			raise Exception("ImageSOP_UID:\n"+ imageSOP_UID+"\n NOT found!")

def roi_region_cut(roi,image_array):
	'''Return cut roi-image stored as Numpy ndarray

	If the diameter of nodule is bigger than DNN_INPUT_NUMBER,we merely cut off 
	its cemteral region so far. 
	'''

	shape = image_array.shape
	xMin,yMin,height,width = rect_region_locate(roi)
	print("LOCATE ROI-IMAGE: X={}, Y={}, HEIGHT={},WIDTH={}".format(xMin,yMin,height,width))
	
	#Whatever the noudule 's size is > 3mm or not
	yCenter = (yMin * 2 + width) // 2
	xCenter = (xMin * 2 + height) // 2

	valid,y1,y2,x1,x2 = boundary_check(xCenter, yCenter, shape)

	if valid:
		print("GENERATE ROI-IMAGE: Y1={}, Y2={}, X1={},X2={}".format(y1,y2,x1,x2))
		# "+1" for slicing			
		return image_array[0,y1:y2 + 1,x1:x2 + 1]
	else:
		return None


DNN_INPUT_SIDE_LEN = 34
def boundary_check(xCenter,yCenter,shape):

	#the shape of Numpy ndaray is indexed as (z,y,x)
	if yCenter - DNN_INPUT_SIDE_LEN // 2 + 1 < 0:
		#Use "//" rather than "/" for floor division
		return False,None,None,None,None
	elif yCenter + DNN_INPUT_SIDE_LEN // 2 > shape[1]:
		return False,None,None,None,None
	elif xCenter - DNN_INPUT_SIDE_LEN // 2 + 1 <0:
		return False,None,None,None,None
	elif xCenter + DNN_INPUT_SIDE_LEN // 2 > shape[2]:
		return False,None,None,None,None
	else:
		return True,yCenter - DNN_INPUT_SIDE_LEN // 2 + 1, \
			yCenter + DNN_INPUT_SIDE_LEN // 2, \
			xCenter - DNN_INPUT_SIDE_LEN // 2 + 1, \
			xCenter + DNN_INPUT_SIDE_LEN // 2 

def rect_region_locate(roi):
	'''Locate the rectangle region of a nodule 's slice
	
	If the diameter of this nodule is > 3mm,return its rectangle boundary
	(xMin,yMin,height ,wdith) with xMin + height := xMax and yMin + width := yMax,respectly
	otherwise return(xCenter,yCenter,0,0) .
	'''
	xCoords = roi.getElementsByTagName("xCoord")
	yCoords = roi.getElementsByTagName("yCoord")
	xData = [int(x.firstChild.data) for x in xCoords]
	yData = [int(y.firstChild.data) for y in yCoords]


	xMin = min(xData)
	yMin = min(yData)
	height = max(xData) - xMin
	width = max(yData) - yMin

	#Note that x-axis towards vertically down 
	return (xMin,yMin,height,width)
