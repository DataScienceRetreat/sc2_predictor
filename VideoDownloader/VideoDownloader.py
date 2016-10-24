#!/usr/bin/env python3
import sys
import subprocess

class VideoDownloader:
	'''
	downloads video in low quality and saves it as mp4 in /data folder
	args: [url, ?start_time, ?end_time]
	file format: ID_TITLE.mp4
	'''
	def __init__(self, args):
		self.args = args

	def download(self):

		if(len(self.args) == 0):
			print('NO URL GIVEN =/ TELL ME WHAT DO DOWNLOAD PLS')
			return -1


		fps = 2
		width = 256
		height = 256
		video_format = "mp4"

		url = str(self.args[0]).strip()

		if(len(self.args) > 1):
			start_time = self.args[1]
		if(len(self.args) > 2):
			end_time = self.args[2]

		bashCommand = """youtube-dl -f bestvideo[height<=?360] --recode-video {video_format} -o data/%(id)s_%(title)s.%(ext)s {url}""".format(
			video_format=video_format,
			url=url)

		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		self.output, self.error = process.communicate()
		
		if(not self.error):
			return 0
		else:
			return -1

if __name__ == "__main__":
   vd = VideoDownloader(sys.argv[1:])
   vd.download()
   
