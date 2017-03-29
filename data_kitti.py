import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
imlist=[]
ylist=[]
listum=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,63,64,65,66,67,79,80,81,82,83]
for i in listum:
	im=cv2.imread("./data_road/training/image_2/um_0000"+str(i)+".png")
	res=cv2.resize(im,(400,400))
	res=res/255.0
	imlist.append(res)
	im2=cv2.imread("./data_road/training/output_gt/um_road_0000"+str(i)+".png")
	res2=np.zeros((800,400))
	for x in range(im2.shape[0]):
		for y in range(im2.shape[1]):
			if (im2[x][y]==[255,0,255]).all():
				res2[x][y]=1

	ylist.append(res2[400:])
	
listumm=[17,18,19,20,21,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,81,82,83,84,85,90,91,92,93,94]
for i in listumm:
	im=cv2.imread("./data_road/training/image_2/umm_0000"+str(i)+".png")
	res=cv2.resize(im,(400,400))
	res=res/255.0
	imlist.append(res)
	im2=cv2.imread("./data_road/training/output_gt/umm_road_0000"+str(i)+".png")
	res2=np.zeros((800,400))
	for x in range(im2.shape[0]):
		for y in range(im2.shape[1]):
			if (im2[x][y]==[255,0,255]).all():
				res2[x][y]=1

	ylist.append(res2[400:])
	
listuu=[11,12,13,14,15,18,19,20,21,22]
for i in listuu:
	im=cv2.imread("./data_road/training/image_2/uu_0000"+str(i)+".png")
	res=cv2.resize(im,(400,400))
	res=res/255.0
	imlist.append(res)
	im2=cv2.imread("./data_road/training/output_gt/uu_road_0000"+str(i)+".png")
	res2=np.zeros((800,400))
	for x in range(im2.shape[0]):
		for y in range(im2.shape[1]):
			if (im2[x][y]==[255,0,255]).all():
				res2[x][y]=1

	ylist.append(res2[400:])
	
	
imlist=np.array(imlist)
ylist=np.array(ylist)
from keras.applications.resnet50 import ResNet50
base_model = VGG16(input_shape=(400,400,3),weights='imagenet', include_top=False)
model = Model(input=base_model.input, output=base_model.get_layer('block5_conv3').output)

np.savez('ylist.npz',ylist=ylist)


all_data=np.load('simple_data_feature_kitti.npz')
feature_list=all_data['imgs']
al_data=np.load('ylist.npz')
ylist=al_data['ylist']
feature_list=feature_list.reshape((-1,5,25,25,512))
ylist=ylist.reshape((-1,5,400,400,1))
hh=np.zeros((1,5,100,100,512))
for i in range(5):
        hh[0][i][40-5*i:65-5*i,40:65]=feature_list[0][i]
        
yy=np.zeros((1,5,3))
for i in range(5):
        yy[0][i][0]=40-5*i
        yy[0][i][1]=40
        yy[0][i][2]=0
        
        
for j in range(120):
        for i in range(5):
                hh[0][i][40-5*i:65-5*i,40:65]=feature_list[j][i]
                sess.run(optimizer,feed_dict={inputs:hh,y_slice:yy,output_y:ylist[j]}) 
