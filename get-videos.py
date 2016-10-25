#!/usr/bin/env python3
import sys
import _thread
from os import listdir
from os.path import isfile, join

from VideoHelper.VideoDownloader import VideoDownloader
from VideoHelper.Video2Images import Video2Images
from VideoHelper.ImageClassifier import ImageClassifier

def get_files_in_dir(dir_path):
	return [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and len(f) > 10]

def get_unique_ids(dir_path):
	return set(f.split('#')[0] for f in get_files_in_dir(dir_path))

def main(args):
	video_path = 'data/video/'
	img_path = 'data/img/'

	if(len(args) < 2):
		print('define input txt file with urls')
		return -1

	unique_video_ids = get_unique_ids(video_path)
	
	f = open(args[1], 'r')
	for line in f:
		if(line[0] == '#') or (len(line) < 5):
			continue
		
		url = line.strip().split(' ')[0]
		if(url.split('?v=')[-1] in unique_video_ids):
			continue

		VideoDownloader(url).download()
	
	f.close()

	print('got all videos in {filename}'.format(filename=args[1]))
	print('generating thumbnail from files')

	video_files = get_files_in_dir(video_path)
	unique_img_ids = get_unique_ids(img_path)

	print('found {n} videos'.format(n = len(video_files)))

	for f in video_files:
		if f.split('#')[0] in unique_img_ids:
			continue

		Video2Images(f).create_thumbnails()

	print('\ndone with thumbnails')
	print('classify thumbnails nooow')
	
	ImageClassifier(get_files_in_dir(img_path))


if __name__ == "__main__":
	main(sys.argv)
