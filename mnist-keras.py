# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 08:47:21 2018

@author: rmohandass
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels) / 255
X_test = X_test.reshape(X_test.shape[0], num_pixels) /255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential([Dense(512, input_shape=(784,)),
                   Activation('sigmoid'),
                   Dense(10),
                   Activation('softmax')
                   ])

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=200, verbose=1, epochs=20, validation_split=0.1)

score = model.evaluate(X_test, y_test, verbose=1)
print('Test Accuracy: ' , score[1])
