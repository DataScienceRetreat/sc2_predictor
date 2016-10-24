#!/usr/bin/env python3
import sys
import _thread

from VideoDownloader import VideoDownloader

def main(args):
	if(len(args) < 2):
		print('define input txt file with urls')
		return -1

	f = open(args[1], 'r')
	for line in f:
		print('current line: {}'.format(line))
		VideoDownloader.VideoDownloader(line.split(' ')).download()

if __name__ == "__main__":
	main(sys.argv)
