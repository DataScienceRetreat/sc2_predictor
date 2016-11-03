# copy files from data/img/
# according to csv file
# to data/train/<class> and data/validation/<class>
import sys
import os
import numpy as np
import pandas as pd
import shutil
import random

from VideoHelper.HelperFunctions import get_files_in_dir

def setup_folders(class_names, path='data/'):
	test_path = path + 'test/'
	validation_path = path + 'validation/'

	for tmp_path in [test_path, validation_path]:
		for class_name in class_names:
			if not os.path.exists(tmp_path + class_name):
				os.makedirs(tmp_path + class_name)


def copy_img(source, class_name, dest_path='data/', test_ratio=0.7):
	dest = ""

	if random.random() > test_ratio:
		dest = dest_path + 'validation/' + class_name
	else:
		dest = dest_path + 'test/' + class_name
	#print("copy from {} to {}".format(source, dest))
	shutil.copy(source, dest)
	
def main(args):

	if(len(args) < 2):
		print('define .csv file and path to image input folder')
		return -1

	filename = args[1]
	folderpath = args[2]
	class_names = ['ingame', 'misc']

	print('filename {}\nfolder {}'.format(filename, folderpath))

	is_file = os.path.isfile(filename)
	if(not is_file):
		return -1

	if not os.path.exists(folderpath):
		return -1

	setup_folders(class_names=class_names)

	df = pd.read_csv(filename, names=['filename','class_name'], dtype={'class_name': np.int8 }, header=1)

	for t in df.itertuples():
		copy_img(folderpath + t.filename + '.png', class_names[t.class_name])

if __name__ == "__main__":
	main(sys.argv)
