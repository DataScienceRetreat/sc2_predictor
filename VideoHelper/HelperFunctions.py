from os import listdir
from os.path import isfile, isdir, join

def get_files_in_dir(dir_path, file_extension=False):
	if file_extension:
		return [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and len(f) > 10]
	else:
		return [f.split('.')[0] for f in listdir(dir_path) if isfile(join(dir_path, f)) and len(f) > 10]

def get_unique_ids(dir_path):
	return set(f.split('#')[0] for f in get_files_in_dir(dir_path))

def path_is_dir(dir_path):
	return isdir(dir_path)