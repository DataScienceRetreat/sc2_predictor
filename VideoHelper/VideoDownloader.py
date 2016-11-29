#!/usr/bin/env python3
import sys
import subprocess

class VideoDownloader:
	'''
	downloads video in low quality and saves it as mp4 in /data folder
	args: [url, ?start_time, ?end_time]
	file format: ID_TITLE.mp4
	'''
	def __init__(self, url, dest='data/video/'):
		self.url = url
		self.dest = dest

	def download(self, verbose=False):
		if verbose:
			print('starting download {}'.format(self.url))
		
		print('.', end="")
		sys.stdout.flush()
		
		if(len(self.url) < 5):
			print('NO URL GIVEN =/ TELL ME WHAT DO DOWNLOAD PLS')
			return -1
		
		#bashCommand = """youtube-dl -f bestvideo[height<=?360] --restrict-filenames -o {dest}%(id)s#%(title)s.%(ext)s {url}""".format(
		bashCommand = """youtube-dl -f bestvideo[height<=?360] --restrict-filenames -o {dest}%(id)s.%(ext)s {url}""".format(
			url=self.url,
			dest=self.dest)

		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		self.output, self.error = process.communicate()

		if verbose:
			print('done with download')

		if(not self.error):
			return 0
		else:
			return -1

if __name__ == "__main__":
   vd = VideoDownloader(sys.argv[1:])
   vd.download()
   
