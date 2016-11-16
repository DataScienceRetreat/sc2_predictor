import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

from scipy.misc import imread, imresize

from VideoHelper.HelperFunctions import get_files_in_dir

img_width = 160
img_height = 90

regression_range = list(range(0, 5))

img_path = '/Users/holger/dev/projects/sc2_predictor/data/img/'
csv_path = '/Users/holger/dev/projects/sc2_predictor/img_class_full.csv'


def csv_to_data(csv_path, target_shape):
    df = pd.read_csv(csv_path)
    X = np.array([imresize(imread(img_path + row['filename'] + '.png'),
                           size=target_shape).transpose(2, 0, 1) for index, row in df.iterrows()], dtype=np.float32)
    X /= 255.
    y = df.iloc[:, 1]
    return (X, y)

print('loading data from csv')

X, y = csv_to_data(csv_path, (img_width, img_height))
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3)

nb_epoch = 15
batch_size = 32

print('building neural network')

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same',  input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

# read csv file with filename and interestingness value
# merge files into huge numpy array
# split into train and test
# fit and evaluate model

print('fitting model')

model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size)
score = model.evaluate(X_test, y_test, batch_size=batch_size)
