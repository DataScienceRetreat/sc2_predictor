import pandas as pd
import numpy as np
import sys
import subprocess
from time import sleep
import datetime
import os.path


class ImageClassifier:
    '''
    class to label images,
    uses quicklook on macOS to have a quick look at images
    loop: image is shown, press space, press class number
    '''

    def __create_command__(self, filename, source):
        return """qlmanage -p '{source}{filename}.png' &> /dev/null""".format(
            # return """open -F -a Preview {source}{filename}.png""".format(
            filename=filename,
            source=source)

    def __init__(self, file_list, source='./data/img/', prev_file='img_class_full.csv'):

        self.source = source
        self.prev_file = prev_file

        has_prev_file = os.path.isfile(prev_file)
        print("has prev file: {}".format(has_prev_file))

        if(has_prev_file):
            self.df_prev = pd.read_csv(self.prev_file, index_col='filename')
            self.file_list = [f for f in file_list if not self.df_prev.index.str.contains(f).any()]
        else:
            self.file_list = [f for f in file_list]

        print("new files: {}".format(len(self.file_list)))

        df = pd.DataFrame(np.zeros(len(self.file_list), dtype=np.int), index=self.file_list,
                          columns=['class'])

        df.index.set_names('filename')

        # check for user abort and save file
        #  - erase files not classified from df
        #  - run append operation after for loop

        n_files = len(file_list)

        for idx, f in enumerate(self.file_list):
            if(idx % 10 == 0):
                print('{0:3.2f}% done  🚀'.format(idx / float(n_files) * 100.0))

            cmd = self.__create_command__(f, source=self.source)
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

            while True:
                file_class = input('enter class: ')

                if(file_class.isdigit()):
                    break

            df.loc[f] = file_class

        print(df.head())

        now = datetime.datetime.now()
        df.to_csv('img_class_new' + str(now.timestamp()) +
                  '.csv', index=True, header=True, index_label='filename', sep=',')

        self.df_prev.append(df).to_csv(
            'img_class_full' + str(now.timestamp()) + '.csv', sep=',')
