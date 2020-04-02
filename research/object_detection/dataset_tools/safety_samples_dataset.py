import tensorflow as tf
import random
import os
import glob
import shutil

# for sampling size percentage. To facilitate growing dataset in view of Active Learning process.
PERCENTAGE = 0.01


# constant
JPEG_IMAGES_DIR = 'JPEGImages/'

flags = tf.app.flags

flags.DEFINE_string('data_dir', 'full-safety-loads/',
                    'full dataset for PASCAL VOC dataset format.')
flags.DEFINE_string('sample_output_dir', 'safety-loads/',
                    'small sample dataset for PASCAL VOC dataset format.')


FLAGS = flags.FLAGS


def main(_):

    data_dir = FLAGS.data_dir
    annotations_dir = os.path.join(data_dir, 'Annotations')

#   label_map_file = os.path.join(FLAGS.label_map_file)
#   label_map_dict = label_map_util.get_label_map_dict(label_map_file)

    annotations_list = glob.glob(os.path.join(annotations_dir, '*.xml'))

    # fixed the file orders and randomness for future sampling,
    # This can help to accumulate previous sampling.
    annotations_list.sort()
    random.seed(42)
    random.shuffle(annotations_list)

    total = len(annotations_list)

    sample_list = annotations_list[0:int(PERCENTAGE*total)]
    print('sample_list size: ', len(sample_list))
#   print('sample_list size: ', sample_list)

    # sample_list to chop full folder name and xml extension
    copy_list = [x.replace(data_dir+'Annotations/',
                           '').replace('.xml', '') for x in sample_list]

    # print('result: ', copy_list)


#    copy to new sample_output_dir
    sample_output_dir = FLAGS.sample_output_dir
    os.makedirs(os.path.join(sample_output_dir, 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(sample_output_dir,
                             JPEG_IMAGES_DIR), exist_ok=True)

    jpeg_output_dir = os.path.join(sample_output_dir, JPEG_IMAGES_DIR)
    jpeg_data_dir = os.path.join(data_dir, JPEG_IMAGES_DIR)

    # print('jpeg_data_dir,', jpeg_data_dir)
    # print('jpeg_output_dir,', jpeg_output_dir)

    for f in copy_list:
        # shutil.copyfile(data_dir+f+'.xml', sample_output_dir+f+'.xml')
        copy_file(os.path.join(data_dir, 'Annotations', f+'.xml'),
                  os.path.join(sample_output_dir, 'Annotations', f+'.xml'))

        copy_file(jpeg_data_dir+f+'.png', jpeg_output_dir+f+'.png')
        copy_file(jpeg_data_dir+f+'.jpeg', jpeg_output_dir+f+'.jpeg')
        copy_file(jpeg_data_dir+f+'.jpg', jpeg_output_dir+f+'.jpg')
        # break  # test 1 case.


def copy_file(source, dest):
    if(os.path.isfile(source)):
        shutil.copyfile(source, dest)


if __name__ == '__main__':
    tf.app.run()
