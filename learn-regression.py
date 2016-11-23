import os, sys
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

import numpy as np
import pandas as pd

from time import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

from scipy.misc import imread, imresize

from VideoHelper.HelperFunctions import get_files_in_dir, path_is_dir

def csv_to_data(csv_path, img_path, target_shape):
    df = pd.read_csv(csv_path)
    X = np.array([imresize(imread(img_path + row['filename'] + '.png'),
                           size=target_shape).transpose(2, 0, 1) for index, row in df.iterrows()], dtype=np.float32)
    X /= 255.
    y = df.iloc[:, 1]
    return (X, y)

def save_model(model, path):
    now = str(time())
    filename = 'SC2_regr_' + now
    model.save(path + 'models/interestingness/' + filename + '.h5')
    plot(model, to_file=path + 'models/interestingness/' + filename + '.png')
    
def get_vgg16_model(img_channels, img_width, img_height, path=None):
    print('building vgg16')

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_channels, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    assert os.path.exists(path), 'model weights not found'
    f = h5py.File(path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('model loaded')

    return model

def main(args):
    img_width = 160
    img_height = 90
    img_channels = 3
    vgg16_path = 'models/vgg16/vgg16_weights.h5'

    regression_range = list(range(0, 5))

    if(len(args) != 3):
        print('give storage folder (containing "data/img/" and "models/" folders) and csv regression file pls')
        return -1

    data_path = args[1]
    if(not path_is_dir(data_path)):
        print('data path is wrooong')
        return -1

    img_path = data_path + 'data/img/'
    if(not path_is_dir(img_path)):
        print('img path is wrooong')
        return -1

    csv_path = args[2]

    print('loading data from csv')

    X, y = csv_to_data(csv_path, img_path, (img_width, img_height))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    nb_epoch = 5
    batch_size = 32

    print('building neural network')

    model = get_vgg16_model(img_channels, img_width, img_height, data_path + vgg16_path)
    print(model.output_shape)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(32))
    top_model.add(Activation('relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1))
    top_model.add(Activation('linear'))

    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.1, momentum=0.9)) #'rmsprop')
    return -1
    print('fitting model')
    model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size)

    save_model(model, data_path)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('mean squared error {}'.format(mse))

if __name__ == "__main__":
    main(sys.argv)
