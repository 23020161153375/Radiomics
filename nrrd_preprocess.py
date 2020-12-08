from PIL import Image
import numpy as np
import nrrd

# nrrd图片读取
# nrrd图片使用nrrd包gitHub中的data数据
nrrd_filename = '../lung/Nodule_0_84.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
print(nrrd_options)
nrrd_image = Image.fromarray(nrrd_data[:,:,100]*100)
nrrd_image.show() # 显示这图片