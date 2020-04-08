import glob
import tensorflow as tf
import sys
import os
import pandas as pd
import shutil
import xml.etree.ElementTree as ET




FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('dir1', None, 'first folder')

tf.app.flags.DEFINE_string('dir2', None, '2nd folder')

tf.app.flags.DEFINE_string('annotation_out', 'annotation_out', 'Merged Annotation output folder')





def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    dir1 = FLAGS.dir1
    dir2 = FLAGS.dir2
    dest_dir = FLAGS.annotation_out

    if dir1 is None or dir2 is None:
        tf.compat.v1.logging.warn('Please provides 2 folders for merging. Output folder is annotation_out by default.')
        tf.compat.v1.logging.warn('--dir1=dir1 --dir2=dir2')
        sys.exit(0)

    dir1_files = glob.glob(os.path.join(dir1,'**', '*.xml'), recursive=True)
    dir2_files = glob.glob(os.path.join(dir2, '**', '*.xml'), recursive=True)

    ## copying no matching files to annotation dir first
    dir1_set = build_set(dir1_files, dir1)
    dir2_set = build_set(dir2_files, dir2)
    os.makedirs(dest_dir, exist_ok=True)
    copy_files_no_matching(dir1_set, dir2_set, dir1, dest_dir )
    copy_files_no_matching(dir2_set, dir1_set, dir2, dest_dir )

    matched_set = dir1_set.intersection(dir2_set)

    for f in matched_set:
        tree = ET.parse(dir2+f)
        objects = tree.findall('object')

        first_tree = ET.parse(dir1+f)
        for o in objects:
            first_tree.getroot().append(o)

        first_tree.write(dest_dir+f)
        tf.logging.debug('Merged to %s', dest_dir+f)        

    tf.logging.debug(' == END ==')

def copy_files_no_matching(dir1_set, dir2_set, dir, dest_dir):
    files_excluded = dir1_set.difference(dir2_set)
    for f in files_excluded:
        tf.logging.debug('copied to %s',dest_dir+f)
        shutil.copyfile(dir+f, dest_dir+f)

def build_set(dir1_files, dir1):
    df = pd.DataFrame(dir1_files, columns=['name'])
    df['name']=df['name'].str.replace(dir1,'')
    name_set = set(df['name'])
    return name_set


    

if __name__ == '__main__':
    main(sys.argv)
