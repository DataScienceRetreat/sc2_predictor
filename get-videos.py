#!/usr/bin/env python3
import sys
import _thread

from VideoHelper.VideoDownloader import VideoDownloader
from VideoHelper.Video2Images import Video2Images
from VideoHelper.HelperFunctions import get_files_in_dir, get_unique_ids

def download_videos(txt_path, img_path, video_path):
	unique_video_ids = get_unique_ids(video_path)
	unique_img_ids = get_unique_ids(img_path)

	f = open(txt_path, 'r')
	for line in f:
		if(line[0] == '#') or (len(line) < 5):
			continue
		
		url = line.strip().split(' ')[0]
		yt_id = url.split('?v=')[-1]
		if(yt_id in unique_video_ids) or (yt_id in unique_img_ids):
			continue

		VideoDownloader(url, video_path).download()
	
	f.close()
	
	print('\ngot all videos in {filename}'.format(filename=txt_path))
	
def make_thumbnails(img_path, video_path):
	print('generating thumbnail from files')

	video_files = get_files_in_dir(video_path, file_extension=True)
	unique_img_ids = get_unique_ids(img_path)

	print('found {n} videos'.format(n = len(video_files)))

	for f in video_files:
		if f.split('#')[0] in unique_img_ids:
			continue

		Video2Images(f, verbose=True).create_thumbnails()
	
	print('\ndone with thumbnails')

def main(args):
	video_path = 'data/video/'
	img_path = 'data/img/'

	if(len(args) < 2):
		print('no input txt file with urls defined')
	else:
		txt_path = args[1]
		download_videos(txt_path, img_path, video_path)
	
	make_thumbnails(img_path, video_path)
	
if __name__ == "__main__":
	main(sys.argv)
