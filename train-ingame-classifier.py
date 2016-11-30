import os
import sys
import glob

import numpy as np
import pandas as pd

from PIL import Image
import requests
import numpy
from io import BytesIO

from time import time

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.layers import Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import CSVLogger, ModelCheckpoint

from sklearn.model_selection import train_test_split

from Helper.HelperFunctions import path_is_file, get_files_in_dir, path_is_dir, files_to_matrix, csv_to_data

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"


def get_filename():
    filename = 'SC2_ingame_' + str(int(time()))
    return filename


def save_model(model, path, filename):
    model.save(path + 'models/ingame-classifier/' + filename + '.h5')


def save_config(path, filename):
    cmd = "cp train-ingame-classifier.py {}_config.txt".format(path + filename)
    os.system(cmd)


def load_photo_from_url(image_url, target_size=None, time_out_image_downloading=1):
    try:
        response = requests.get(image_url, timeout=time_out_image_downloading)
    except requests.exceptions.RequestException:
        response = requests.get(
            image_url, timeout=time_out_image_downloading * 10)
    try:
        image = Image.open(BytesIO(response.content))
    except IOError:
        raise IOError('Error with image form url {}'.format(image_url))
    if target_size is not None:
        image = image.resize((target_size[1], target_size[0]))
    img_array = img_to_array(image) / 255.0
    img_array = img_array.reshape((1,) + img_array.shape)
    return img_array


def load_img_from_file(img_path, target_size=None, time_out_image_downloading=1):
    image = load_img(img_path)
    if target_size is not None:
        image = image.resize((target_size[1], target_size[0]))
    img_array = img_to_array(image) / 255.0
    img_array = img_array.reshape((1,) + img_array.shape)
    return img_array


def get_sample_count(path, classes):
    count = 0
    for name in class_names:
        count += len(glob.glob1(path + name + '/', '*.png'))
    return count


def get_model(img_channels, img_width, img_height, dropout=0.5):

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(
        img_channels, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def main(args):

    if(len(args) < 3):
        print('give storage folder (containing "data/ingame", "data/misc", "models/" and "logs/" folders) and csv classification file pls')
        return -1

    base_path = args[1]
    data_path = base_path + "data/"
    model_path = base_path + "models/"

    if(not path_is_dir(data_path)):
        print('data path is wrooong')
        return -1

    csv_path = args[2]
    log_path = base_path + 'logs/'

    img_channels = 3
    img_width = 160
    img_height = 90

    class_names = ['ingame', 'misc']

    nb_epoch = 40
    batch_size = 16
    
    # TODO: check for filename, class in header of csv 
    print('load ingame images')
    X_ingame, y_ingame = csv_to_data(csv_path, data_path + 'ingame/', 'class', (img_width, img_height), limit=100)
    print('load misc images')
    X_misc, y_misc = csv_to_data(csv_path, data_path + 'misc/', 'class', (img_width, img_height), limit=100)
    
    assert X_ingame.shape[0] != 0

    X = np.append(X_ingame, X_misc, axis=0)
    y = np.append(y_ingame, y_misc, axis=0)

    new_indices = np.random.permutation(X.shape[0])
    np.take(X, new_indices, axis=0, out=X)
    np.take(y, new_indices, axis=0, out=y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    nb_train_samples = X_train.shape[0]
    nb_validation_samples = X_test.shape[0]
    
    print('nb_train_samples {}'.format(nb_train_samples))
    print('nb_validation_samples {}'.format(nb_validation_samples))

    model = get_model(img_channels, img_width, img_height)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
    )

    train_datagen.fit(X_train)

    test_datagen = ImageDataGenerator()
    test_datagen.fit(X_test)

    filename = get_filename()
    csv_logger=CSVLogger(log_path + filename + '.log')
    model_logger = ModelCheckpoint(model_path + 'ingame-classifier/' + filename + '.h5', 
        save_best_only=True, save_weights_only=False, mode='auto')

    model.fit_generator(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=train_datagen.flow(X_test, y_test),
        nb_val_samples=nb_validation_samples,
        callbacks=[csv_logger, model_logger])

    #    save_model(model, model_path, filename)

    img_internet = load_photo_from_url(r"http://asset-9.soupcdn.com/asset/16161/1072_950a_649.png", (img_width, img_height))
    array_to_img(img_internet[0])

    tst_img_path = data_path + 'ingame/' + '0M3rGNWZTlk#_00001.png'
    img = load_img_from_file(tst_img_path, (img_width, img_height))
    #array_to_img(img[0])

    print('{} should be 0'.format(model.predict(img)))
    print('{} should be 1'.format(model.predict(img_internet)))

    from Helper.NotificationSender import NotificationSender
    NotificationSender('IngameClassifier').notify('training NN is done 🚀😍')

if __name__ == "__main__":
    main(sys.argv)

