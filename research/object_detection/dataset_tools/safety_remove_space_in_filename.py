'''

credit from https://stackoverflow.com/questions/7469374/renaming-file-names-containing-spaces
'''


import os

path  = os.getcwd()
filenames = os.listdir(path)
for filename in filenames:
    os.rename(os.path.join(path, filename), os.path.join(path, filename.replace(' ', '-')))