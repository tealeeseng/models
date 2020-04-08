
'''

remove space in filename, credit from https://stackoverflow.com/questions/7469374/renaming-file-names-containing-spaces
'''

import glob
import tensorflow as tf
import sys
import os
import pandas as pd
import shutil
import xml.etree.ElementTree as ET




FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('data_dir', None, 'dataset folder')


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    data_dir = FLAGS.data_dir

    if data_dir is None:
        tf.logging.warn('Please provides dataset folder.')
        tf.logging.warn('--data_dir=dir')
        sys.exit(0)

    remove_space_in_filenames(data_dir)        

    total_count = remove_space_in_filename_inside_annotations(data_dir)
        # break    

    tf.compat.v1.logging.info(' %s files processed. == END ==', total_count)

def remove_space_in_filenames(data_dir):
    filenames = glob.glob(os.path.join(data_dir,'**','*'), recursive=True)
    # print('filenames:',filenames)


    for filename in filenames:
        new_filename = filename.replace(' ', '-').replace('(','-').replace(')','-')
        os.rename( filename, new_filename)
        # print('new_filename:',new_filename)

def remove_space_in_filename_inside_annotations(dir):
    dir_files = glob.glob(os.path.join(dir,'**', '*.xml'), recursive=True)
    dir_files.sort()
    ## copying no matching files to annotation dir first

    i=0
    for f in dir_files:
        tree = ET.parse(f)
        filename_node = tree.find('filename')

        filename_node.text = filename_node.text.replace(' ','-').replace('(','-').replace(')','-')
        # tf.logging.debug('filename_node, %s', filename_node.text)    
        tree.write(f)
        i = i+1
    return i

    

if __name__ == '__main__':
    main(sys.argv)
