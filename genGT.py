########################
# @Resnick Xing
# DRION-DB annotation to groundtruth
########################
import pandas as pd
import glob
import cv2
import numpy as np

# make dir for groundtruth image
def mkdir(path):
	import os
	path = path.strip()
	path = path.rstrip("\\")
	isExists = os.path.exists(path)
	if not isExists:
		print(path + 'create successfully!')
		os.makedirs(path)
		return True
	else:
		print(path + 'already exists')
		return False

def txt2gt(imageAnnotationFileList,expert1,expert2,showImage=False):
	count=1
	for txtFilePath in imageAnnotationFileList:
		imageName='image_'+(((txtFilePath.split('\\')[-1]).split('.')[0]).split('_')[-1])+'.jpg'
		data=pd.read_table(txtFilePath,header=None)
		rows=data.shape[0]
		coord=[]
		for line in range(rows):
			coord_str=data.values[line][0]
			coord_x, coord_y=coord_str.split(',')
			coord.append([int(float(coord_x)),int(float(coord_y))])
		coord=np.array([coord],np.int32)

		image=cv2.imread('.\\images\\'+imageName)
		groundtruth=np.zeros_like(image[:,:,0])
		cv2.fillPoly(groundtruth,coord,255)
		cv2.polylines(image,coord,1,255)
		if showImage:
			cv2.imshow(imageName,image)
			cv2.waitKey(800)
			cv2.destroyWindow(imageName)
		if count<=110:
			print("processing expert1 annotation:",count,"/100")
			cv2.imwrite(expert1+imageName,groundtruth)
		else:
			print("processing expert2 annotation:", count-110, "/100")
			cv2.imwrite(expert2 + imageName, groundtruth)
		count=count+1

if __name__=="__main__":
	expert1='./groundtruth/expert1/'
	expert2='./groundtruth/expert2/'
	showImage=False

	mkdir(expert1)
	mkdir(expert2)
	imageAnnotationFileList = glob.glob('.\\experts_anotation\\*.txt')
	txt2gt(imageAnnotationFileList,expert1,expert2,showImage)


