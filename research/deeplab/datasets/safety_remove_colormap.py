"""Removes the color map from segmentation annotations.

Removes the color map from the ground truth segmentation annotations and save
the results to output_dir.
"""
import glob
import os.path
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import os


def sample_code():
    filename = './safety/SegmentationClass/frame_000000.png'
    test_filename = './safety/r-frame_000000.png'
    DATASET_FOLDER = 'safety'

    # filename = './pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png'
    # test_filename = './pascal_voc_seg/VOCdevkit/VOC2012/r-2007_000032.png'
    color_df_f = build_color_df(DATASET_FOLDER)

    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_maps = []

    for r in color_df_f.iterrows():
        ci = r[0]
        c_class = r[1][0]
        color_code = r[1][1]
        color_codes = [int(x) for x in color_code.split(',')]
        # print(color_codes)

        # image = Image.open(filename)

        # print('black count: ', np.sum(np.all(image==[0,0,0],axis=2)))
        gray_maps.append(np.all(image == color_codes, axis=2) * ci)
        print(c_class, ': ', np.sum(np.all(image == color_codes, axis=2)))

    gray_img = np.sum(gray_maps, axis=0)
    print(gray_img)

    pil_image = Image.fromarray(gray_img)

    with tf.gfile.Open(test_filename, mode='w') as f:
        pil_image.save(f, 'PNG')


def _remove_colormap_rgb(filename, color_df):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_maps = []

    i = 0
    total = 0
    for r in color_df.iterrows():
        ci = r[0]
        c_class = r[1][0]
        color_code = r[1][1]
        color_codes = [int(x) for x in color_code.split(',')]
        # print(color_codes)

        # image = Image.open(filename)

        # print('black count: ', np.sum(np.all(image==[0,0,0],axis=2)))
        gray_maps.append(np.all(image == color_codes, axis=2) * ci)
        value = np.sum(np.all(image == color_codes, axis=2))
        tf.compat.v1.logging.info('class: %s, count: %s, weight: %s, value:%s', c_class, value, i, value * i)
        total = total + value * i
        i = i + 1

    gray_img = np.sum(gray_maps, axis=0)
    tf.compat.v1.logging.info('gray_img.shape: %s', gray_img.shape)
    tf.compat.v1.logging.info(gray_img)
    tf.compat.v1.logging.info('total value:%s check:%s', total, np.sum(gray_img))

    return gray_img


def build_color_df(DATASET_FOLDER):
    color_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'labelmap.txt'),
                           sep=':', header=0, names=['label', 'color', 'part', 'action'],
                           skip_blank_lines=True)
    color_df_f = color_df.sort_values(by='color').reset_index(drop=True)
    color_df_f.to_csv(os.path.join(DATASET_FOLDER, 'gray_map.txt'))
    return color_df_f


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('original_gt_folder',
                           './VOCdevkit/VOC2012/SegmentationClass',
                           'Original ground truth annotations.')

tf.app.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')

tf.app.flags.DEFINE_string('output_dir',
                           './VOCdevkit/VOC2012/SegmentationClassRaw',
                           'folder to save modified ground truth annotations.')


def _remove_colormap(filename):
    """Removes the color map from the annotation.

    Args:
      filename: Ground truth annotation filename.

    Returns:
      Annotation without color map.
    """
    return np.array(Image.open(filename))


def _save_annotation(annotation, filename):
    """Saves the annotation as png file.

    Args:
      annotation: Segmentation annotation.
      filename: Output filename.
    """
    pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
    with tf.io.gfile.GFile(filename, mode='w') as f:
        pil_image.save(f, 'PNG')


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

    # Create the output directory if not exists.
    if not tf.io.gfile.isdir(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    annotations = glob.glob(os.path.join(FLAGS.original_gt_folder,
                                         '*.' + FLAGS.segmentation_format))
    color_df = build_color_df('safety')

    for annotation in annotations:
        raw_annotation = _remove_colormap_rgb(annotation, color_df)
        filename = os.path.splitext(os.path.basename(annotation))[0]

        _save_annotation(raw_annotation,
                         os.path.join(
                             FLAGS.output_dir,
                             filename + '.' + FLAGS.segmentation_format))
        # break


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
