import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

img_width = 90
img_height = 160

regression_range = range(0, 5)
train_data_dir = '/Users/holger/dev/projects/sc2_predictor/data/test/'
validation_data_dir = '/Users/holger/dev/projects/sc2_predictor/data/validation/'
img_path = '/Users/holger/dev/projects/sc2_predictor/data/img/'

nb_epoch = 15
batch_size = 32

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('tanh'))
model.add(Dense(64, 1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size)
score = model.evaluate(X_test, y_test, batch_size=batch_size)