import os

import dataGenerator

def prepare_all_data(raw_data_root,intermedia_data_root):
	'''Get intermedia data(unorganized)

	The parameter intermedia_data_root is output path root.
	'''

	#THe hierarchy of TICA data organization is:
	#Collection Name -> Subject -> Study -> Series -> slices(DICOM files)
	#The raw_data_root is the path to our collection name("LIDC-IDRI") directory.

	f_label = open("Label.txt", 'a')
	f_erased = open("Erased.txt", 'a')
	for subject_dir_names in os.listdir(raw_data_root):
		subject_dir = os.path.join(raw_data_root,subject_dir_names)
		if os.path.isdir(subject_dir):
			for study_dir_name in os.listdir(subject_dir):
				study_dir = os.path.join(subject_dir,study_dir_name)
				if os.path.isdir(study_dir):
					for series_dir_name in os.listdir(study_dir):
						series_dir = os.path.join(study_dir,series_dir_name)
						if os.path.isdir(series_dir) and is_CT_dir(series_dir):
							print("CT Dir:",series_dir)
							try:
								save_name_complete,save_name_erased,character_list=dataGenerator.gen_roi_images(series_dir, intermedia_data_root)
								print(save_name_complete+'_cut.npy'+' '+str(character_list)+'\n')
								print(save_name_erased+'.npy'+' '+str(character_list)+'\n')
								f_label.write(save_name_complete+'_cut.npy'+' '+str(character_list)+'\n')
								f_erased.write(save_name_erased+'.npy'+' '+str(character_list)+'\n')
							except:
								break
	# f_label.close()
	# f_erased.close()

def is_CT_dir(case_path):
	'''To clissify CT and DX(X-ray) directorys 

	According to incomplete statistics, a DX derectory has 3 files at most
	while a CT directory has more than a hundred of files. :-)
	'''
	if(os.path.exists(case_path) and os.path.isdir(case_path)):
		if(len(os.listdir(case_path)) > 10):
			return True
	return False

if __name__ == '__main__':
	raw_data_root = r"D:\cancer\gan\lung\LIDC_dataset\LIDC-IDRI"
	intermedia_data_root = r"D:\cancer\gan\lung"

	prepare_all_data(raw_data_root, intermedia_data_root)
