import cv2
import tensorflow as tf
import os
import numpy as np

CATAGORIES = ["Ok", "Not Ok"]

cap = cv2.VideoCapture(0)



#def prepare(img):
	#img_size = 128
	#img_array = cv2.imread(img, cv2.IMREAD_COLOR)
	#new_array = cv2.resize(img_array, (img_size, img_size))
	#new_array = new_array.reshape(-1, img_size, img_size, 3)

model = tf.keras.models.load_model("protoFix3.model")

while True:
	success, image = cap.read()
	#img1 = cv2.imread(img, cv2.IMREAD_COLOR)
	image = cv2.resize(image,(512,512))

	ycbcr = image.copy()
	ycbcr = cv2.cvtColor(ycbcr, cv2.COLOR_BGR2YCR_CB)

	lower_y = np.array([0,0,118])
	upper_y = np.array([255,255,255])

	mask = cv2.inRange(ycbcr, lower_y, upper_y)
	result = cv2.bitwise_and(ycbcr, ycbcr, mask = mask)

	result2 = result.reshape(-1, 512, 512,3)
	
	prediction = model.predict([result2])
	print("Prediksi	:", CATAGORIES[int(prediction[0][0])], prediction)
	cv2.imshow("Img",result)


	#
	#print("Prediksi	:", CATAGORIES[int(prediction[0][0])])

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


