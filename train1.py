import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np

start = time.time()

X =  pickle.load(open("train_XFix4.pickle","rb"))
y =  pickle.load(open("label_yFix4.pickle","rb"))

#X = X/255.0

y = np.array(y)


#Name = "Modelku{}".format(int(time.time()))

#TB = TensorBoard(log_dir = 'logs/{}'.format(Name))

model = Sequential()
#1 Conv Layer
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
#Max Pooling
model.add(MaxPooling2D(pool_size=(7,7)))

#2 Conv Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
#Max Pooling
model.add(MaxPooling2D(pool_size=(7,7)))

#Flatten
model.add(Flatten())

#1 Fully Connected
model.add(Dense(128, input_shape = X.shape[1:]))
model.add(Activation("relu"))
#Dropout
model.add(Dropout(0.5))

#2 Fully Connected
model.add(Dense(128))
model.add(Activation("relu"))
#Dropout
model.add(Dropout(0.5))

#Output Layer
model.add(Dense(1))
model.add(Activation("sigmoid"))

#np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit(X,y, batch_size = 1, epochs = 10, validation_split = 0.3)

model.save('protoFix15.model')
model.summary()

print("Sukses")
end = time.time()
timeku = end - start
print("Waktu Training	:", timeku, "Detik")