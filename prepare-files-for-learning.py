# copy files from data/img/
# according to csv file
# to data/train/<class> and data/validation/<class>
import sys
import os
import numpy as np
import pandas as pd
import shutil
import random

from VideoHelper.HelperFunctions import get_files_in_dir, path_is_dir

def setup_folders(class_names, path='data/', noTestnoValid=False):
	test_path = path + 'test/'
	validation_path = path + 'validation/'

	if noTestnoValid:
		test_path, validation = path, path

	for tmp_path in [test_path, validation_path]:
		for class_name in class_names:
			if not os.path.exists(tmp_path + class_name):
				os.makedirs(tmp_path + class_name)


def copy_img(source, class_name, dest_path='data/', test_ratio=0.7, noTestnoValid=False, dry=False, verbose=False):
	dest = ""
	rand = random.random()
	
	if noTestnoValid:
		dest = dest_path + class_name
	else:
		if rand > test_ratio:
			dest = dest_path + 'validation/' + class_name
		else:
			dest = dest_path + 'test/' + class_name
	
	if verbose:
		print("copy from {} to {}".format(source, dest))
	
	if not(dry):
		shutil.copy(source, dest)
	
def main(args):

	if(len(args) != 4):
		print('define paths to image input folder, destination folder and .csv file')
		return -1

	img_path = args[1]
	dest_path = args[2]
	filename = args[3]

	if (not(path_is_dir(img_path)) or not(path_is_dir(dest_path))):
		print('given folders not available to load or save data')
		return -1
	
	class_names = ['ingame', 'misc']	
	print('filename {}\nfolder {}'.format(filename, img_path))

	is_file = os.path.isfile(filename)
	if(not is_file):
		print('given csv file not available')
		return -1

	setup_folders(class_names=class_names)

	df = pd.read_csv(filename, names=['filename','class_name'], dtype={'class_name': np.int8 }, header=1)
	print('start copying {} files'.format(df.shape[0]))
	for t in df.itertuples():
		copy_img(img_path + t.filename + '.png', class_names[t.class_name], dest_path)
	
	print('done')

if __name__ == "__main__":
	main(sys.argv)
