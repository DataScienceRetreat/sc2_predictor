import os
import sys

import numpy as np
import pandas as pd

from Helper.HelperFunctions import get_files_in_dir, path_is_dir, files_to_matrix
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

def download_video(video_id, video_path, data_path):
    return 0

def create_empty_df(folder, path):

    files = get_files_in_dir(path + folder)
    df = pd.DataFrame(index=files)

    return df

def make_predictions(df, data, model, pred_name):
    """predicts with given data and model -> stored in pred_name column
    """

    df[pred_name] = np.zeros(df.shape[0])

    for index, row in df.iterrows():
        pred = model.predict(np.array([data[df.index.get_loc(index)]]))[0]
        df.loc[index][pred_name] = pred

    return df

def criteria_fullfilled(row, dependencies, criteria):
    """give col names in dependencies
    and values in criteria that given cols should NOT be equal to
    """
    for idx, d in enumerate(dependencies):
        if row[d] == criteria[idx]:
            return False

    return True

def make_predictions_with_criteria(df, data, model, pred_name, dependencies=[], criteria=[]):
    """predicts with given data and model -> stored in pred_name column,
    only if criteria are fulfilled (values in criteria should not be met)
    """
    assert len(dependencies) == len(criteria)

    df[pred_name] = np.ones(df.shape[0]) * -1.

    for index, row in df.iterrows():
        if (criteria_fullfilled(row, dependencies, criteria)):
            pred = model.predict(np.array([data[df.index.get_loc(index)]]))[0]
            df.loc[index][pred_name] = pred

    return df

def download_video(video_id, video_url, data_path, ext='mp4'):
    tmp_folder = data_path + video_id
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    VideoDownloader(video_url, dest=tmp_folder).download()

    video_path = tmp_folder + video_id + '.' + ext
    Video2Images(video_path, fps=1, source=tmp_folder, dest=tmp_folder).create_thumbnails()

def get_filename():
    filename = 'SC2_regr_' + str(time())
    return filename

def main(args):
    if len(args) < 5:
        print('some params are missing, sry')
        print('give yt link or folder with images, ingame model,')
        print('interestingness model, tmp/ path, fps (default=0.5)')
        return -1

    video_path = args[1]
    ingame_model_str = args[2]
    interestingness_model_str = args[3]
    data_path = args[4]
    fps = 1
    video_id = ''

    img_width = 160
    img_height = 90
    img_channels = 3
    shape = (img_width, img_height)

    if len(args) > 5:
        fps = float(args[5])

    if not(string_is_link(video_path)) and not(path_is_dir(video_path)):
        print('give youtube link or folder with images')
        return -1

    if string_is_link(video_path):
        video_id = video_path.split('?v=')[-1]
        download_video(video_id, video_path, data_path)
    else:
        video_id = video_path.split('/')[-1]

    df = create_empty_df(video_id, data_path)

    X = files_to_matrix(df, data_path + video_id + '/', shape, column='index')
    
    print(df.head())
    print(X.shape)

    ingame_model = load_model(ingame_model_str)
    df = make_predictions(df, X, ingame_model, 'ingame')

    print('after predictions shape {}'.format(df.shape))
    print(df.head())

    interestingness_model = load_model(interestingness_model_str)
    df = make_predictions_with_criteria(df, X, interestingness_model, 'interestingness', ['ingame'], [1])
    
    df.to_csv(out_filename, index=False)

if __name__ == "__main__":
    main(sys.argv)
