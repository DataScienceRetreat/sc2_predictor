#!/usr/bin/env python3
import sys
import _thread
from VideoHelper.ImageClassifier import ImageClassifier
from VideoHelper.HelperFunctions import get_files_in_dir, path_is_dir

def main(args):

	img_path = args[1]
	if(not path_is_dir(img_path)):
		return -1

	img_files = get_files_in_dir(img_path, file_extension=False)
	print('classify thumbnails nooow')
	print('0 -> in-game, 1 -> everything out of game')
	ImageClassifier(img_files)
	
if __name__ == "__main__":
	main(sys.argv)