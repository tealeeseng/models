# 
# ==============================================================================

"""
prepare train.txt and val.txt after segmentation annotation.

"""

import glob
import os.path
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import os



FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

tf.app.flags.DEFINE_string('training_percentage', "0.8", 'percentage of training.')



def build_files_list(dataset_folder, segmentation_images, training_percentage):
    images_filenames = pd.DataFrame(segmentation_images)
    images_filenames = images_filenames[0].apply(lambda x: x.split(os.path.sep)[2])
    images_filenames = images_filenames.str.replace('.png', '')
    images_filenames = images_filenames.sample(frac=1)
    total = len(images_filenames)
    train_size = int(total * training_percentage)
    train = images_filenames[:train_size]
    val = images_filenames[train_size:]
    trainval = images_filenames
    folder = os.path.join(dataset_folder, 'ImageSets', 'Segmentation')
    train.to_csv(os.path.join(folder, 'train.txt'), index=False, header=False)
    val.to_csv(os.path.join(folder, 'val.txt'), index=False, header=False)
    trainval.to_csv(os.path.join(folder, 'trainval.txt'), index=False, header=False)

    default_file = os.path.join(folder, 'default.txt')
    if os.path.isfile(default_file):
        os.remove(default_file)

    tf.logging.debug('train: %s Val: %s', len(train),len(val))



def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    dataset_folder = FLAGS.dataset_dir

    SEGMENTATION_FOLDER = os.path.join(dataset_folder , 'SegmentationClass')

    segmentation_images = glob.glob(SEGMENTATION_FOLDER + os.path.sep + '*.png')
    tf.logging.debug('images: %s ', segmentation_images)

    build_files_list(dataset_folder, segmentation_images, float(FLAGS.training_percentage))





if __name__ == '__main__':
    tf.compat.v1.app.run()

#
# annotation = np.array(image)
# # np.savetxt('a.txt', annotation)
# # annotation[600]
#
# annotation_8 = annotation.astype(dtype=np.uint8)
#
# print(( annotation_8))
#
#
# pil_image = Image.fromarray(annotation_8)
#
# with tf.gfile.Open(test_filename, mode='w') as f:
#     pil_image.save(f, 'PNG')
