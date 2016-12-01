import os
import sys
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

import numpy as np
import pandas as pd

from time import time

import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import h5py

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, SpatialDropout2D
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model
import keras.backend as K

from Helper.HelperFunctions import path_is_file, get_files_in_dir, path_is_dir, files_to_matrix, _obtain_input_shape, csv_to_data

def get_filename():
    filename = 'SC2_regr_' + str(int(time()))
    return filename


def get_old_config(path, model_path):
    old_config = path + model_path.split('/')[-1].split('.h5')[0] + ''
    return old_config


def copy_config(path, model_path, filename):
    old_config = get_old_config(path, model_path)

    if path_is_file(old_config):
        cmd = 'cp {old} {new}_config.txt'.format(
            old=old_config, new=path + filename)
        print('copying old config with cmd:\n{}'.format(cmd))
        os.system(cmd)
    else:
        cmd = "echo 'Could not find old config for this model' >> {new}_config.txt".format(
            new=path + filename)
        print('could not find old config for this model')
        os.system(cmd)


def save_config(path, filename):
    cmd = "cp learn-regression.py {}_config.txt".format(path + filename)
    os.system(cmd)


def save_model(model, path, filename):
    model.save(path + 'models/interestingness/' + filename + '.h5')
    # from keras.utils.visualize_util import plot
    # plot(model, to_file=path + 'models/interestingness/' + filename + '.png')


def VGG16(input_shape=None, model_path='models'):
    """adapted from keras repo at
    https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
    """
    # Determine proper input shape
    TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
    TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu',
              border_mode='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Convolution2D(128, 3, 3, activation='relu',
              border_mode='same', name='block2_conv1'))
    model.add(Convolution2D(128, 3, 3, activation='relu',
              border_mode='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    model.add(Convolution2D(256, 3, 3, activation='relu',
              border_mode='same', name='block3_conv1'))
    model.add(Convolution2D(256, 3, 3, activation='relu',
              border_mode='same', name='block3_conv2'))
    model.add(Convolution2D(256, 3, 3, activation='relu',
              border_mode='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    model.add(Convolution2D(512, 3, 3, activation='relu',
              border_mode='same', name='block4_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu',
              border_mode='same', name='block4_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu',
              border_mode='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    model.add(Convolution2D(512, 3, 3, activation='relu',
              border_mode='same', name='block5_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu',
              border_mode='same', name='block5_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu',
              border_mode='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    if K.image_dim_ordering() == 'th':
        weights_path=get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
                                TH_WEIGHTS_PATH_NO_TOP,
                                cache_subdir=model_path)

        model.load_weights(weights_path)
    else:
        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir=model_path)
        model.load_weights(weights_path)

        if K.backend() == 'theano':
            convert_all_kernels_in_model(model)

        print('model loaded')

        if K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image dimension ordering convention '
                          '(`image_dim_ordering="th"`). '
                          'For best performance, set '
                          '`image_dim_ordering="tf"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
            convert_all_kernels_in_model(model)

    print('before pop')
    print(len(model.layers))
    model.pop()
    model.pop()
    model.pop()
    model.pop()
    print(len(model.layers)) 

    for layer in model.layers[:25]:
        layer.trainable=False

    return model

def get_model(shape, dropout=0.5, path=None):
    print('building neural network')

    model=Sequential()

    # model.add(Convolution2D(96, 7, 7, border_mode='same',
    #                         input_shape=shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(SpatialDropout2D(dropout))

    # model.add(Convolution2D(128, 5, 5, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(SpatialDropout2D(dropout))

    # model.add(Convolution2D(256, 3, 3, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(SpatialDropout2D(dropout))

    # model.add(Convolution2D(512, 3, 3, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(SpatialDropout2D(dropout))

    model.add(Convolution2D(1024, 3, 3, border_mode='same', input_shape=shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(dropout))

    model.add(Flatten())#input_shape=shape))
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
    img_width=160
    img_height=90
    img_channels=3
    img_shape=(img_channels, img_width, img_height)

    regression_range=list(range(0, 5))

    if(len(args) < 3):
        print('give storage folder (containing "data/ingame/" and "models/" folders) and csv regression file pls')
        return -1

    base_path=args[1]
    if(not path_is_dir(base_path)):
        print('data path is wrooong')
        return -1

    img_path=base_path + 'data/ingame/'
    if(not path_is_dir(img_path)):
        print('img path is wrooong')
        return -1

    csv_path=args[2]
    log_path=base_path + 'logs/'

    model_path=''
    if(len(args) > 3):
        model_path=args[3]

    print('loading data from csv')
    X, y=csv_to_data(csv_path, img_path, 'interestingness', (img_width, img_height))
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3)

    nb_epoch=120
    batch_size=32

    nb_train_samples=X_train.shape[0]
    nb_validation_samples=X_test.shape[0]

    print('nb_train_samples {}'.format(nb_train_samples))
    print('nb_validation_samples {}'.format(nb_validation_samples))

    train_datagen=ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            rotation_range=10.,
    )

    train_datagen.fit(X_train)

    test_datagen=ImageDataGenerator()
    test_datagen.fit(X_test)

    filename=get_filename()
    csv_logger=CSVLogger(log_path + filename + '.log')
    model_logger = ModelCheckpoint(base_path + 'models/interestingness/' + filename + '.h5', 
        save_best_only=True, save_weights_only=False, verbose=True, mode='auto')

    if model_path:
        print('loading model from {}'.format(model_path))
        top_model=get_model(shape=(img_channels, img_width, img_height))
        top_model=load_model(model_path)
        copy_config(log_path, model_path, filename)
    else:
        model=VGG16(input_shape=img_shape, model_path=base_path + 'models/')
        top_model=get_model(shape=model.output_shape[1:])
        model.add(top_model)
        save_config(log_path, filename)

    model.compile(loss='mean_squared_error',
          optimizer=SGD(lr=0.1, momentum=0.9))#'adam') #'rmsprop') #

    print('model compiled')
    # print('fitting model')

    # model.fit(X_train, y_train, nb_epoch=nb_epoch, callbacks=[csv_logger, model_logger],
              # batch_size=batch_size, validation_data=(X_test, y_test))

    model.fit_generator(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=test_datagen.flow(X_test, y_test),
        nb_val_samples=nb_validation_samples,
        callbacks=[csv_logger, model_logger])

    # save_model(model, base_path, filename)

    y_pred=model.predict(X_test)
    mse=mean_squared_error(y_test, y_pred)
    print('mean squared error {}'.format(mse))

    from Helper.NotificationSender import NotificationSender
    NotificationSender('RegressionLearner').notify(
        'training NN is done 🚀😍\nmse: {}'.format(mse))

if __name__ == "__main__":
    main(sys.argv)
