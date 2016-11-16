#!/usr/bin/env python3
import sys
import _thread
from VideoHelper.ImageClassifier import ImageClassifier
from VideoHelper.HelperFunctions import path_is_dir

def main(args):
	
	if(len(args) < 2):
		print('give image folder (and optional existing csv class file) pls')
		return -1

	img_path = args[1]
	if(not path_is_dir(img_path)):
		return -1

	if len(args) == 3:
		class_file = args[2] 
		ImageClassifier(img_path, prev_file=class_file)
	else:
		ImageClassifier(img_path)
	
if __name__ == "__main__":
	main(sys.argv)