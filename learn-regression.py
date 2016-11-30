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
from keras.layers.core import Dense, Dropout, Activation, SpatialDropout2D
from keras.layers import Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from keras.models import load_model

from Helper.HelperFunctions import path_is_file, get_files_in_dir, path_is_dir, files_to_matrix

def csv_to_data(csv_path, img_path, target_shape):
    df = pd.read_csv(csv_path)

    X = files_to_matrix(df, img_path, target_shape)

    X /= 255.
    y = np.array([row['interestingness'] for index, row in df.iterrows()
        if os.path.isfile(img_path + row['filename'] + '.png')])
    # y = df.iloc[:, 1]
    print('X shape {}'.format(X.shape))

    return (X, y)

def get_filename():
    filename = 'SC2_regr_' + str(int(time()))
    return filename

def get_old_config(path, model_path):
    old_config = path + model_path.split('/')[-1].split('.h5')[0] + ''
    return old_config

def copy_config(path, model_path, filename):
    old_config = get_old_config(path, model_path)

    if path_is_file(old_config):
        cmd = 'cp {old} {new}_config.txt'.format(old=old_config, new=path + filename)
        print('copying old config with cmd:\n{}'.format(cmd))
        os.system(cmd)
    else:
        cmd = "echo 'Could not find old config for this model' >> {new}_config.txt".format(new=path + filename)
        print('could not find old config for this model')
        os.system(cmd)

def save_config(path, filename):
    cmd = "cp learn-regression.py {}_config.txt".format(path + filename)
    os.system(cmd)

def save_model(model, path, filename):
    model.save(path + 'models/interestingness/' + filename + '.h5')
    # from keras.utils.visualize_util import plot
    # plot(model, to_file=path + 'models/interestingness/' + filename + '.png')

def get_model(img_channels, img_width, img_height, dropout=0.5, path=None):
    print('building neural network')

    model = Sequential()

 
    model.add(Convolution2D(96, 7, 7, border_mode='same',
                            input_shape=(img_channels, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(dropout))

    model.add(Convolution2D(128, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(dropout))

    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(dropout))

    # model.add(Convolution2D(512, 3, 3, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(SpatialDropout2D(dropout))

    # model.add(Convolution2D(1024, 3, 3, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(SpatialDropout2D(dropout))
 
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('linear'))

    return model


def main(args):
    img_width = 160
    img_height = 90
    img_channels = 3

    regression_range = list(range(0, 5))

    if(len(args) < 3):
        print('give storage folder (containing "data/ingame/" and "models/" folders) and csv regression file pls')
        return -1

    data_path = args[1]
    if(not path_is_dir(data_path)):
        print('data path is wrooong')
        return -1

    img_path = data_path + 'data/ingame/'
    if(not path_is_dir(img_path)):
        print('img path is wrooong')
        return -1

    csv_path = args[2]
    log_path = data_path + 'logs/'

    model_path = ''
    if(len(args) > 3):
        model_path = args[3]

    print('loading data from csv')
    # TODO: check csv for filename, interestingness in header
    X, y = csv_to_data(csv_path, img_path, (img_width, img_height))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    nb_epoch = 80
    batch_size = 128

    filename = get_filename()
    csv_logger = CSVLogger(log_path + filename + '.log')

    if model_path:
        print('loading model from {}'.format(model_path))
        model = get_model(img_channels, img_width, img_height)
        model = load_model(model_path)
        copy_config(log_path, model_path, filename)
    else:
        model = get_model(img_channels, img_width, img_height)
        save_config(log_path, filename)

    model.compile(loss='mean_squared_error',
          optimizer='adam') #'rmsprop') #SGD(lr=0.001, momentum=0.9))

    print('model compiled')
    print('fitting model')

    model.fit(X_train, y_train, nb_epoch=nb_epoch, callbacks=[csv_logger], 
              batch_size=batch_size, validation_data=(X_test, y_test))
    
    save_model(model, data_path, filename)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('mean squared error {}'.format(mse))

    from Helper.NotificationSender import NotificationSender
    NotificationSender('RegressionLearner').notify('training NN is done 🚀😍\nmse: {}'.format(mse))

if __name__ == "__main__":
    main(sys.argv)
