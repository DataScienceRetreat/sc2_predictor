#!/usr/bin/env python3
import sys
import subprocess

class Video2Images:
	'''
	splits video into images
	needs video file, fps, destination_folder, ?start_time, ?end_time
	'''

	def __init__(self, filepath, fps=1, source='data/video/', dest='data/img/'):
		self.filepath = filepath
		self.fps = fps
		self.source = source
		self.dest = dest
		self.title = filepath.split('/')[-1].split('#')[0]

	def create_thumbnails(self):
		bashCommand = """ffmpeg -i {source}{filepath} -loglevel panic -vf fps={fps}/60 {dest}{title}#_%05d.png""".format(
			source=self.source,
			filepath=self.filepath,
			fps=self.fps,
			title=self.title,
			dest=self.dest)
		
		print('.', end="")
		sys.stdout.flush()
		
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		self.output, self.error = process.communicate()
		
		if(not self.error):
			return 0
		else:
			return -1
