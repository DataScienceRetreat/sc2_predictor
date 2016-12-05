import os
import sys
from time import time

import numpy as np
import pandas as pd

from Helper.HelperFunctions import get_files_in_dir, path_is_dir, files_to_matrix, path_is_file
from Helper.VideoDownloader import VideoDownloader
from Helper.Video2Images import Video2Images

from keras.models import load_model

# script to analyze interestingness of a video
# input:
#  - youtube link or folder with images,
#  - ingame classifier model
#  - interestingness regression model
#  - fps (default=0.5)
# output: time series with interestingness

# steps
# 1. download video to data/tmp/<file id>/ folder
#  - youtube-dl -> video file
#  - ffmpeg -> frames
#  - pandas df with filename, ingame, interestingness
# 2. run ingame model on all images
#  - write resut to pandas df
# 3. run interestingness model on ingame files
#  - write results to pandas df
# 4. output csv from pandas df


def string_is_link(str):
    return '://www.' in str

def create_filename_df(folder_path, ext='.png'):

    files = get_files_in_dir(folder_path + '/', file_extension=True)
    files = [f.split('.')[0] for f in files if ext in f]

    df = pd.DataFrame(index=files)

    return df


def make_predictions(df, data, model, pred_name, binary=False, decimal_precision=4):
    """predicts with given data and model -> stored in pred_name column
    """
    df[pred_name] = np.zeros(df.shape[0])

    for index, row in df.iterrows():
        # print('index {} row {}'.format(index, row))
        # print('loc {}'.format(df.index.get_loc(index)))

        pred = model.predict(np.array([data[df.index.get_loc(index)]]))[0]
        if binary:
            df.loc[index][pred_name] = int(round(pred[0], 0))
        else:
            df.loc[index][pred_name] = round(pred[0], decimal_precision)

    return df


def criteria_fullfilled(row, dependencies, criteria):
    """give col names in dependencies
    and values in criteria that given cols should NOT be equal to
    """
    for idx, d in enumerate(dependencies):
        if row[d] == criteria[idx]:
            return False

    return True

def setup_folder(data_path, video_id):
    tmp_folder = data_path + video_id + '/'

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    return tmp_folder

def make_predictions_with_criteria(df, data, model, pred_name, dependencies=[], criteria=[], decimal_precision=4):
    """predicts with given data and model -> stored in pred_name column,
    only if criteria are fulfilled (values in criteria should not be met)
    """
    assert len(dependencies) == len(criteria)

    df[pred_name] = np.ones(df.shape[0]) * -1.

    for index, row in df.iterrows():
        if (criteria_fullfilled(row, dependencies, criteria)):
            pred = model.predict(np.array([data[df.index.get_loc(index)]]))[0]
            df.loc[index][pred_name] = round(pred[0], decimal_precision)

    return df


def download_video(video_id, video_url, data_path, ext='mp4'):
    tmp_folder = setup_folder(data_path, video_id)

    filename = video_id + '.' + ext

    if not path_is_file(tmp_folder + filename):
        VideoDownloader(video_url, dest=tmp_folder, verbose=False).download()

    return tmp_folder + filename

def video_to_images(video_id, video_path, data_path, ext='mp4', isdir=True):
    tmp_folder = setup_folder(data_path, video_id)

    if isdir and len(get_files_in_dir(tmp_folder)) > 5:
        return 0
    else:
        Video2Images(video_path, fps=1, source='',
                     dest=tmp_folder, verbose=False).create_thumbnails()


def load_images(df, folder, shape):
    X = files_to_matrix(df, folder, shape, column='index')
    X /= 255.

    return X

def main(args):
    if len(args) < 5:
        print('some params are missing, sry')
        print('give yt link or folder with images, ingame model,')
        print('interestingness model, tmp/ path, fps (default=0.5)')
        return -1

    video_location = args[1]
    ingame_model_path = args[2]
    interestingness_model_path = args[3]
    data_path = args[4]
    fps = 1
    video_id = ''

    img_width = 160
    img_height = 90
    img_channels = 3
    shape = (img_width, img_height)

    if len(args) > 5:
        fps = float(args[5])

    if string_is_link(video_location):
        video_id = video_location.split('?v=')[-1]

        video_path = download_video(video_id, video_location, data_path)
        video_to_images(video_id, video_path, data_path)
    else:
        video_id = video_location.split('/')[-1].split('.')[0]
        if path_is_file(video_location) and '.mp4' in video_location:
            video_to_images(video_id, video_location, data_path)

    working_folder = data_path + video_id + '/'

    df = create_filename_df(working_folder)
    X = load_images(df, working_folder, shape)

    ingame_model = load_model(ingame_model_path)
    df = make_predictions(df, X, ingame_model, 'ingame', binary=True)

    interestingness_model = load_model(interestingness_model_path)
    df = make_predictions_with_criteria(
        df, X, interestingness_model, 'interestingness', ['ingame'], [1])

    csv_path = working_folder + 'analysis_' + str(int(time())) + '.csv'
    
    if df.shape[0] > 1:
        df.to_csv(csv_path, index_label='filename', index=True)

if __name__ == "__main__":
    main(sys.argv)
