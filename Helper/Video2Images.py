#!/usr/bin/env python3
import sys
import subprocess


class Video2Images:
    '''
    splits video into images
    needs video file, fps, destination_folder, ?start_time, ?end_time
    '''

    def __init__(self, filename, fps=1, source='data/video/', dest='data/img/', verbose=False):
        self.filename = filename
        self.fps = fps
        self.source = source
        self.dest = dest
        self.title = filename.split('/')[-1].split('.')[0]
        self.verbose = verbose

    def create_thumbnails(self):
        bashCommand = """ffmpeg -i {source}{filename} -loglevel panic -vf fps={fps} {dest}{title}#_%05d.png""".format(
            source=self.source,
            filename=self.filename,
            fps=self.fps/60.,
            title=self.title,
            dest=self.dest)

        print('.', end="")
        sys.stdout.flush()

        if self.verbose:
            print('execute cmd:\n{}'.format(bashCommand))

        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        self.output, self.error = process.communicate()

        if self.verbose:
            print(self.output)

        if(not self.error):
            return 0
        else:
            return -1
