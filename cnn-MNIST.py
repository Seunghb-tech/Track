#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Input
from keras.models import Model, Sequential, load_model
from keras.utils import to_categorical

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)

plt.imshow(X_train[0], cmap='binary')
print(Y_train[0])

#(60000, 28, 28) -> (60000, 28, 28, 1)
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

X_train = X_train/255.0
X_test = X_test/255.0

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', strides=1, activation='relu', input_shape=(28,28,1,)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=36, kernel_size=(5,5), padding='valid', strides=1, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=200, epochs=1, validation_split=0.2)

score = model.evaluate(X_test, Y_test)
print(score)


'''
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 24, 24, 16)        416       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 16)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 36)          14436     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2304)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               295040    
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290 
'''
l1 = model.get_layer('conv2d_1')
# l1.get_weights()   
print(l1.get_weights()[0].shape) # --> (5, 5, 1, 16)

# (5, 5, 1, 16) --> (16, 5, 5, 1)

def plot_weight(w):
    w_min = np.min(w)
    w_max = np.max(w)
    
    num_grid = math.ceil(math.sqrt(w.shape[3]))  # 16
    
    fig, axis = plt.subplots(num_grid, num_grid)
    
    for i, ax in enumerate(axis.flat):
        if i < w.shape[3] :
            img = w[:,:,0,i]
            ax.imshow(img,vmin=w_min, vmax=w_max )
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

l1 = model.get_layer('conv2d_1')
w1 = l1.get_weights()[0]   
plot_weight(w1)

l2 = model.get_layer('conv2d_2')
w2 = l2.get_weights()[0]   
plot_weight(w2)

temp_model = Model(inputs=model.get_layer('conv2d_1').input, outputs=model.get_layer('conv2d_1').output)
output = temp_model.predict(X_test)
# print(output.shape)

def plot_output(output):
    num_grid = math.ceil(math.sqrt(output.shape[3]))  # 16
    
    fig, axis = plt.subplots(num_grid, num_grid)
    
    for i, ax in enumerate(axis.flat):
        if i < output.shape[3] :
            img = output[0,:,:,i]
            ax.imshow(img, cmap='binary' )
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    

plot_output(output)

temp_model = Model(inputs=model.get_layer('conv2d_1').input, outputs=model.get_layer('conv2d_2').output)
output = temp_model.predict(X_test)
# print(output.shape)

plot_output(output)

