import os, sys
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

from scipy.misc import imread, imresize

from VideoHelper.HelperFunctions import get_files_in_dir, path_is_dir

def csv_to_data(csv_path, img_path, target_shape):
    df = pd.read_csv(csv_path)
    X = np.array([imresize(imread(img_path + row['filename'] + '.png'),
                           size=target_shape).transpose(2, 0, 1) for index, row in df.iterrows()], dtype=np.float32)
    X /= 255.
    y = df.iloc[:, 1]
    return (X, y)

def main(args):
    img_width = 160
    img_height = 90

    regression_range = list(range(0, 5))

    if(len(args) != 3):
        print('give image folder and csv regression file pls')
        return -1

    img_path = args[1]
    if(not path_is_dir(img_path)):
        return -1

    csv_path = args[2]

    print('loading data from csv')

    X, y = csv_to_data(csv_path, img_path, (img_width, img_height))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print('X_train shape {}'.format(X_train.shape))
    print('y_train shape {}'.format(y_train.shape))

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
    #score = model.evaluate(X_test, y_test, batch_size=batch_size)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('mean squared error {}'.format(mse))

if __name__ == "__main__":
    main(sys.argv)
