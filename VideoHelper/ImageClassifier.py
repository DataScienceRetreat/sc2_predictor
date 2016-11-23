import pandas as pd
import numpy as np
import sys
import subprocess
from time import sleep
import datetime
import os.path

from VideoHelper.HelperFunctions import get_files_in_dir, path_is_dir

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

    def __init__(self, img_path, new_filename ='img_class', prev_file='img_class_full.csv'):

        print('classify thumbnails')

        files = get_files_in_dir(img_path, file_extension=False)

        self.source = img_path
        self.prev_file = prev_file
        self.new_filename = new_filename

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

        # check for user abort and save file
        #  - erase files not classified from df
        #  - run append operation after for loop

        n_files = len(self.file_list)
        
        print("new files: {}".format(n_files))
        answer = input("do all or enter number [all|<number>]: ")
        
        if answer.strip() != 'all':
            n_files = int(answer)

        df = pd.DataFrame(np.zeros(n_files, dtype=np.int), index=self.file_list[:n_files],
                              columns=new_column)

        df.index.set_names('filename')

        for idx, f in enumerate(self.file_list[:n_files]):
            if(idx % 10 == 0):
                print('{0:3.2f}% done  🚀'.format(idx / float(n_files) * 100.0))

            cmd = self.__create_command__(f, source=self.source)
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

            while True:
                file_class = input('enter class: ')

                if(file_class.isdigit()):
                    break

            df.loc[f] = file_class

        now = datetime.datetime.now()
        df.to_csv(self.new_filename + '_new' + str(now.timestamp()) +
                  '.csv', index=True, header=True, index_label='filename', sep=',')

        self.df_prev.append(df).to_csv(
            self.new_filename + '_full' + str(now.timestamp()) + '.csv', sep=',')
