import os
import sys
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

import numpy as np
import pandas as pd

from time import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
# from keras.utils.visualize_util import plot

from scipy.misc import imread, imresize

from VideoHelper.HelperFunctions import get_files_in_dir, path_is_dir
from NotificationSender import NotificationSender


def csv_to_data(csv_path, img_path, target_shape):
    df = pd.read_csv(csv_path)

    X = np.array([imresize(imread(img_path + row['filename'] + '.png'),
                           size=target_shape).transpose(2, 0, 1)
                  for index, row in df.iterrows() if os.path.isfile(img_path + row['filename'] + '.png')
                  ], dtype=np.float32)
    X /= 255.
    y = np.array([row['interestingness'] for index, row in df.iterrows()
        if os.path.isfile(img_path + row['filename'] + '.png')])
    # y = df.iloc[:, 1]

    return (X, y)


def save_model(model, path):
    now = str(time())
    filename = 'SC2_regr_' + now
    model.save(path + 'models/interestingness/' + filename + '.h5')
    # plot(model, to_file=path + 'models/interestingness/' + filename + '.png')


def get_model(img_channels, img_width, img_height, path=None):
    print('building neural network')

    model = Sequential()

    model.add(Convolution2D(48, 7, 7, border_mode='same',
                            input_shape=(img_channels, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(96, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('linear'))
    print('model loaded')

    return model


def main(args):
    img_width = 160
    img_height = 90
    img_channels = 3

    regression_range = list(range(0, 5))

    if(len(args) != 3):
        print('give storage folder (containing "data/img/" and "models/" folders) and csv regression file pls')
        return -1

    data_path = args[1]
    if(not path_is_dir(data_path)):
        print('data path is wrooong')
        return -1

    img_path = data_path + 'data/img/ingame/'
    if(not path_is_dir(img_path)):
        print('img path is wrooong')
        return -1

    csv_path = args[2]

    print('loading data from csv')

    X, y = csv_to_data(csv_path, img_path, (img_width, img_height))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    nb_epoch = 100
    batch_size = 32

    model = get_model(img_channels, img_width, img_height)
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=0.1, momentum=0.9))

    print('fitting model')
    model.fit(X_train, y_train, nb_epoch=nb_epoch,
              batch_size=batch_size, validation_data=(X_test, y_test))

    save_model(model, data_path)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('mean squared error {}'.format(mse))

    NotificationSender('RegressionLearner').notify('training NN is done 🚀😍')

if __name__ == "__main__":
    main(sys.argv)
