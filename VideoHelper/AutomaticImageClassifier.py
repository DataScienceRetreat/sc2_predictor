import pandas as pd
import numpy as np
import sys
import datetime
import os.path

from scipy.misc import imread, imresize

from VideoHelper.HelperFunctions import get_files_in_dir, path_is_dir

from keras.models import load_model

class AutomaticImageClassifier:
    '''
    class to label images,
    uses neural network to automatically classify images
    '''

    def __init__(self, model_path, img_path, new_filename ='img_class', prev_file='img_class_full.csv', target_shape=(160, 90)):

        print('loading AutomaticImageClassifier')

        files = get_files_in_dir(img_path, file_extension=False)

        self.source = img_path
        self.prev_file = prev_file
        self.new_filename = new_filename
        self.model = load_model(model_path)

        has_prev_file = os.path.isfile(prev_file)
        print("has prev file: {}".format(has_prev_file))

        if(has_prev_file):
            self.df_prev = pd.read_csv(self.prev_file, index_col='filename')
            if self.df_prev.shape[0] == 0:
                self.file_list = [f for f in files]
            else:
                self.file_list = [
                    f for f in files if not self.df_prev.index.str.contains(f).any()]
        else:
            self.file_list = [f for f in files]

        if(len(self.file_list) == 0):
            print("no new files to be classified")
            return
        
        new_columns = ""
        if 'interestingness' in self.prev_file:
            new_column = ['interestingness']
            self.new_filename = 'img_interestingness'
        else:
            new_column = ['class']

        n_files = len(self.file_list)
        
        print("new files: {}".format(n_files))
        df = pd.DataFrame(np.zeros(n_files, dtype=np.int), index=self.file_list[:n_files],
                              columns=new_column)

        df.index.set_names('filename')

        print('start to classify') 
        for idx, f in enumerate(self.file_list):
            if(idx % 10 == 0):
                print('\r\r\r{0:3.2f}% done  🚀'.format(idx / float(n_files) * 100.0), end='')
                sys.stdout.flush()
        
            img = np.array(imresize(imread(img_path + f + '.png'),
                       size=target_shape).transpose(2, 0, 1), dtype=np.float32)
            img /= 255.
            file_class = self.model.predict(np.array([img]))[0]
            np.round(file_class, decimals=2, out=file_class)
            # print('class for {} is {}'.format(f, file_class[0]))
            df.loc[f] = int(file_class[0])
        
        print('\nsaving files')

        now = datetime.datetime.now()
        df.to_csv(self.new_filename + '_new' + str(now.timestamp()) +
                  '.csv', index=True, header=True, index_label='filename', sep=',')

        self.df_prev.append(df).to_csv(
            self.new_filename + '_full' + str(now.timestamp()) + '.csv', index_label='filename', sep=',')
