from os import listdir
from os.path import isfile, isdir, join
from scipy.misc import imread, imresize

import numpy as np
import pandas as pd

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