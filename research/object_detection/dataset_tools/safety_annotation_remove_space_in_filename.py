import glob
import tensorflow as tf
import sys
import os
import pandas as pd
import shutil
import xml.etree.ElementTree as ET




FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('dir', None, 'annotation folder')


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    dir = FLAGS.dir

    if dir is None:
        tf.logging.warn('Please provides folder')
        tf.logging.warn('--dir=dir')
        sys.exit(0)

    dir_files = glob.glob(os.path.join(dir,'**', '*.xml'), recursive=True)
    dir_files.sort()
    ## copying no matching files to annotation dir first

    i=0
    for f in dir_files:
        tree = ET.parse(f)
        filename_node = tree.find('filename')

        filename_node.text = filename_node.text.replace(' ','-')
        # tf.logging.debug('filename_node, %s', filename_node.text)    
        tree.write(f)
        i = i+1
        # break    

    tf.logging.info(' %s files processed. == END ==', i)

    

if __name__ == '__main__':
    main(sys.argv)
