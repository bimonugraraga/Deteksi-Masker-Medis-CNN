#import tensorflow as tfs
import numpy as np

import os
import cv2
import pickle
import random
#from tqdm import tqdm
#import csv


DATADIR = "Masker3"#"tangan/Arrow"
CATAGORIES = ["Ok", "Not Ok"]

#for category in CATAGORIES :
#	path = os.path.join(DATADIR,category)
#	for img in os.listdir(path):
#		img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
#		plt.imshow(img_array, cmap = "gray")
#		plt.show()
#		break
#	break
IMG_SIZE = 512
#new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#plt.imshow(new_array, cmap = 'gray')
#plt.show()

####################################################
training_data =[]
def create_training_data():
	for category in CATAGORIES :
		path = os.path.join(DATADIR,category)
		class_num = CATAGORIES.index(category)
		for img in os.listdir(path):
			try :
				image = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
				#new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				image = cv2.resize(image,(512,512))
			
				ycbcr = image.copy()
				ycbcr = cv2.cvtColor(ycbcr, cv2.COLOR_BGR2YCR_CB)
				#hsv = image.coy()
				#hsv = cv2.cvtColor(hsv,cv2.COLOR_BGR2HSV)

				lower_y = np.array([0, 0, 118])
				upper_y = np.array([255, 255, 255])

				#lower_h = np.array([0, 0, 13])
				#upper_h = np.array([255, 191, 255])

				mask = cv2.inRange(ycbcr, lower_y, upper_y)
				result = cv2.bitwise_and(ycbcr,ycbcr,mask = mask)

				#mask1 = cv2.inRange(hsv, lower_h, upper_h)
				#result1 = cv2.bitwise_and(result,result,mask = mask1)
				#cv2.imshow("ok", result)
				#cv2.waitKey(0)
				result = np.array(result).flatten()
				result = np.average(result)
				training_data.append([result, class_num])
				#plt.imshow(img_array, cmap = "gray")
				#plt.show()
			except Exception as e :
				pass
create_training_data()
print(len(training_data))
#print(training_data)
random.shuffle(training_data)

train_X = []
label_y = []

for features, label in training_data :
	train_X.append(features)
	label_y.append(label)

#train_X = np.array(train_X).reshape(-1, 512, 512, 3)

#pickle_out = open("train_X.pickle", "wb")
pickle_out = open("train_XFix5.pickle", "wb")
pickle.dump(train_X, pickle_out)
pickle_out.close()

#pickle_out = open("label_y.pickle", "wb")
pickle_out = open("label_yFix5.pickle", "wb")
pickle.dump(label_y, pickle_out)
pickle_out.close()

#print(training_data)

print("sukses")

