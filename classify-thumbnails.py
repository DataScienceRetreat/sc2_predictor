#!/usr/bin/env python3
import sys
import _thread
from VideoHelper.ManualImageClassifier import ManualImageClassifier
from VideoHelper.AutomaticImageClassifier import AutomaticImageClassifier
from VideoHelper.HelperFunctions import path_is_dir

def main(args):
	
	if(len(args) < 3):
		print('give mode [manual|automatic], image folder (and model for automatic mode and optional existing csv class file) pls')
		return -1

	option = args[1]
	img_path = args[2]


	if(not path_is_dir(img_path)):
		print('image folder "{}" not available'.format(img_path))
		return -1

	if 'manual' in str(option):	
		if len(args) == 4:
			prev_file = args[3] 
			ManualImageClassifier(img_path, prev_file=prev_file)
		else:
			ManualImageClassifier(img_path)
	elif 'auto' in str(option) and len(args) > 3:
		model_path = args[3]

		if len(args) == 4:
			AutomaticImageClassifier(model_path, img_path)
		elif len(args) == 5:
			prev_file = args[4]
			AutomaticImageClassifier(model_path, img_path, prev_file=prev_file)
	else:
		print('check the params again, something didnt work out sry')

if __name__ == "__main__":
	main(sys.argv)