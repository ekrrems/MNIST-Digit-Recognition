from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Dense,Flatten
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img,array_to_img
import matplotlib.pyplot as plt 
import pandas as pd
from glob import glob
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import pickle

# Exploring the dataset

(X_train,y_train),(X_test,y_test) = mnist.load_data()

print("""
Shape of X_train:{}
Shape of y_train:{}
Shape of X_test: {}
Shape of y_test:{}

""".format(X_train.shape,y_train.shape,X_test.shape,y_test.shape))

plt.imshow(X_train[0])
plt.show()
print(y_train[0])

x = img_to_array(X_train[0])
print(x.shape)

a=pd.DataFrame(y_train)
print(a[0].unique())
print(a[0].value_counts())

numofclass= len(a[0].unique())
print(numofclass)

# Building the Convolutional Neural Network Models

model = Sequential()

model.add(Conv2D(32,(5,5),input_shape = x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024)) #What Dense Layer Does?
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(numofclass))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

# TRAINING MODEL

batch_size =32

train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range = 0.4,
                                  zoom_range=0.3)
test_datagen = ImageDataGenerator(rescale=1./255)


X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],X_test.shape[2],1))


train_generator = train_datagen.flow(X_train,y_train,
                                     batch_size=batch_size)
test_generator = test_datagen.flow(X_test,y_test)


hist=model.fit_generator(train_generator,
                   steps_per_epoch=1600//batch_size,
                   epochs=90,
                   validation_data=test_generator,
                   validation_steps = 800//batch_size)


pickle_out = open("mnist_digit_model.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()
