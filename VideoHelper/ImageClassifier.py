import pandas as pd
import numpy as np
import sys
import subprocess
from time import sleep

class ImageClassifier:
	'''
	class to label images,
	uses quicklook on macOS to have a quick look at images
	loop: image is shown, press space, press class number
	'''
	
	def __create_command__(self, filename, source):
		return """qlmanage -p '{source}{filename}.png' &> /dev/null""".format(
		#return """open -F -a Preview {source}{filename}.png""".format(
			filename=filename,
			source=source)

	def __init__(self, file_list, source='./data/img/'):
		self.file_list = [f.split('.')[0] for f in file_list]
		self.source = source

		print(self.file_list[:5])
		df = pd.DataFrame(np.zeros(len(self.file_list), dtype=np.int), index= self.file_list,
			columns=['class'])

		df.head()

		for f in self.file_list[:5]:
			cmd = self.__create_command__(f, source=self.source)
			print(cmd)

			process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
			
			#sleep(1)
			#process.kill()
			#process.terminate()

			while True:
				file_class = input('enter class: ')
				# print(file_class)
				
				if(file_class.isdigit()):
					break

			# df.loc[f] = file_class

		df.to_csv('img_class_full.csv', sep=',')

