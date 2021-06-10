import cv2
import numpy as np
import tensorflow as tf
import glob

CATAGORIES = ["Medis", "Non Medis"]
path = glob.glob("setmeter/*.png")
model = tf.keras.models.load_model("protoFix12.model")
print("Ok")
uji =[]
number = 0
num = 1
for file in path:
	image = cv2.imread(file,1)
	image = cv2.resize(image,(512,512))

	ycbcr = image.copy()
	ycbcr = cv2.cvtColor(ycbcr, cv2.COLOR_BGR2YCR_CB)

	lower_y = np.array([0,0,118])
	upper_y = np.array([255,255,255])

	mask = cv2.inRange(ycbcr, lower_y, upper_y)
	result = cv2.bitwise_and(ycbcr, ycbcr, mask = mask)

	#result = np.array(result)
	result = result.reshape(-1, 512, 512,3)
	uji.append([result])
	cv2.imshow("Ok",image)
	cv2.waitKey(0)
	prediction = model.predict([result])
	print("predict", num,":", CATAGORIES[int(prediction[0][0])])
	number+=1
	num+=1
print("Jumlah Data : ", len(uji))
print("Data 0,5 meter")
print("Sukses")