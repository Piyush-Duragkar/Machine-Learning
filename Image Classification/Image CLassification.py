import tensorflow as tf
from tensorflow import keras
import tensorflow.keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
# %matplotlib inline
import numpy as np

#Importing the dataset from keras datasets

from keras.datasets import mnist

(train_data,train_target),(test_data,test_target)=mnist.load_data()

#Checking wether it is the same dataset

print(train_data.shape)
print(test_data.shape)
print(train_target.shape)
print(test_target.shape)

plt.imshow(train_data[0],cmap='gray')
plt.show()

#Categorical Conversion of the data

from keras.utils import np_utils

new_train_target=np_utils.to_categorical(train_target)
new_test_target=np_utils.to_categorical(test_target)

#printing the first 30 values of converted data
print(train_target[:30])
print(new_train_target[:30])

new_train_data=train_data/255
new_test_data=test_data/255

from keras.models import Sequential
from keras.layers import Dense,Flatten

model=Sequential()

model.add(Flatten(input_shape=(28,28)))

model.add(Dense(500,activation='relu'))
model.add(Dense(500,activation='relu'))


model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(new_train_data,new_train_target,epochs=250)

plt.plot(model.history.history['loss'])
plt.xlabel('EPOCHS')
plt.ylabel('LOSS')
plt.show()

plt.plot(model.history.history['accuracy'])
plt.xlabel('EPOCHS')
plt.ylabel('ACCURACY')
plt.show()

model.evaluate(new_test_data,new_test_target)

from keras.models import Sequential
from keras.layers import Dense,Flatten

model=Sequential()

model.add(Flatten(input_shape=(28,28)))

model.add(Dense(500,activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(Dense(500,activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit(new_train_data,new_train_target,epochs=250)

plt.plot(model.history.history['loss'])
plt.xlabel('EPOCHS')
plt.ylabel('LOSS')
plt.show()

plt.plot(model.history.history['accuracy'])
plt.xlabel('EPOCHS')
plt.ylabel('ACCURACY')
plt.show()

model.evaluate(new_test_data,new_test_target)

