from os import listdir
from os.path import isfile, isdir, join
from scipy.misc import imread, imresize

import numpy as np
import pandas as pd
import warnings

def path_is_file(path):
    return isfile(path)


def get_files_in_dir(dir_path, file_extension=False):
    if file_extension:
        return [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and len(f) > 10]
    else:
        return [f.split('.')[0] for f in listdir(dir_path) if isfile(join(dir_path, f)) and len(f) > 10]


def get_unique_ids(dir_path):
    return set(f.split('#')[0] for f in get_files_in_dir(dir_path))


def path_is_dir(dir_path):
    return isdir(dir_path)


def image_to_matrix(path, shape):
    return imresize(imread(path), size=shape).transpose(2, 0, 1)


def files_to_matrix(df, img_path, shape, column='filename', ending='.png'):
    """converts files given in a dataframe to a 4D matrix
    expects filenames in dataframe under column (default='filename')
    expects images in path with ending (default='.png') 
    """
    if column == 'index':
        return np.array([image_to_matrix(img_path + str(index) + ending, shape)
                         for index, row in df.iterrows() if isfile(img_path + str(index) + ending)
                         ], dtype=np.float32)
    else:
        return np.array([image_to_matrix(img_path + row[column] + ending, shape)
                         for index, row in df.iterrows() if isfile(img_path + row[column] + ending)
                         ], dtype=np.float32)


def _obtain_input_shape(input_shape, default_size, min_size, dim_ordering):
    if dim_ordering == 'th':
        default_shape = (3, default_size, default_size)
    else:
        default_shape = (default_size, default_size, 3)

    if dim_ordering == 'th':
        if input_shape is not None:
            if len(input_shape) != 3:
                raise ValueError(
                    '`input_shape` must be a tuple of three integers.')
            if input_shape[0] != 3:
                raise ValueError('The input must have 3 channels; got '
                                 '`input_shape=' + str(input_shape) + '`')
            if ((input_shape[1] is not None and input_shape[1] < min_size) or
                    (input_shape[2] is not None and input_shape[2] < min_size)):
                raise ValueError('Input size must be at least ' +
                                 str(min_size) + 'x' +
                                 str(min_size) + ', got '
                                 '`input_shape=' + str(input_shape) + '`')
        else:
            input_shape = (3, None, None)
    else:
        if input_shape is not None:
            if len(input_shape) != 3:
                raise ValueError(
                    '`input_shape` must be a tuple of three integers.')
            if input_shape[-1] != 3:
                raise ValueError('The input must have 3 channels; got '
                                 '`input_shape=' + str(input_shape) + '`')
            if ((input_shape[0] is not None and input_shape[0] < min_size) or
                    (input_shape[1] is not None and input_shape[1] < min_size)):
                raise ValueError('Input size must be at least ' +
                                 str(min_size) + 'x' +
                                 str(min_size) + ', got '
                                 '`input_shape=' + str(input_shape) + '`')
        else:
            input_shape = (None, None, 3)
    return input_shape
