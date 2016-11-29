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
        response = requests.get(image_url, timeout=time_out_image_downloading * 10)
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

def main(args):

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

    img_width = 160
    img_height = 90

    class_names = ['ingame', 'misc']
    train_data_dir = '/Users/holger/dev/projects/sc2_predictor/data/train/'
    validation_data_dir = '/Users/holger/dev/projects/sc2_predictor/data/test/'
    img_path = '/Users/holger/dev/projects/sc2_predictor/data/img/'

    def get_sample_count(path, classes):
        count = 0
        for name in class_names:
            count += len(glob.glob1(path + name + '/', '*.png'))
        return count

    nb_train_samples = get_sample_count(train_data_dir, class_names)
    nb_validation_samples = get_sample_count(validation_data_dir, class_names)
    
    nb_epoch = 80
    batch_size = 32

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
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
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

    filename = get_filename()
    save_model(model, data_path, filename)

    img_internet = load_photo_from_url(r"http://asset-9.soupcdn.com/asset/16161/1072_950a_649.png", (img_width, img_height))
    array_to_img(img_internet[0])

    tst_img_path = img_path + '0M3rGNWZTlk#_00001.png'
    img = load_img_from_file(tst_img_path, (img_width, img_height))
    array_to_img(img[0])

    print(model.predict(img)) # should be 0
    print(model.predict(img_internet)) # should be 1

if __name__ == "__main__":
    main(sys.argv)
