# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.
Example usage:
    python object_detection/dataset_tools/create_helmet_tf_record.py \
        --data_dir=/home/user/VoTToutput \
        --output_dir=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import glob
import random

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
# flags.DEFINE_string('data_dir', 'D:\\trainJPG', 'Root directory to raw PASCAL VOC dataset.')
# flags.DEFINE_string('set', 'train', 'Convert training set or validation set.')
# flags.DEFINE_string('annotations_dir', 'object_detection\\construction_dataset\\testdataset\\labels\\',
#                     '(Relative) path to annotations directory.')
# flags.DEFINE_string('output_dir', 'object_detection\\construction_dataset\\testdataset', 'Path to output TFRecord')
# flags.DEFINE_string('label_map_file', 'object_detection\\construction_Dataset\\construction_label_map.pbtxt',
#                     'Path to label map proto')
# flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
#                      'difficult instances')
# flags.DEFINE_string('imageset_path', 'object_detection\\construction_dataset\\testdataset\\ImageSets', 'path to image sets')

flags.DEFINE_string('data_dir', 'safety-loads/', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'JPEGImages', 'Convert training set or validation set.')
# flags.DEFINE_string('annotations_dir', 'safety-loads/Annotations/',
                    # '(Relative) path to annotations directory.')
flags.DEFINE_string('output_dir', 'safety-loads/', 'Path to output TFRecord')
flags.DEFINE_string('label_map_file', 'safety-loads/safety_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_string('imageset_path', 'safety-loads/ImageSets', 'path to image sets')


FLAGS = flags.FLAGS

# SETS = ['train', 'val']

def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.
  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.
  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.
  Returns:
    example: The converted tf.Example.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(data['filename'])
  full_path = os.path.join(dataset_directory, image_subdirectory , img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    # print('image_name, ', img_path)
    # print('image format,',image.format)
    buf_io = io.BytesIO()
    image.save(buf_io, format='JPEG')
    newimage = PIL.Image.open(buf_io)
    
    # nnImage = Image.open(io.BytesIO(buf_io.getvalue()))
    # print('nnImage format,',nnImage.format)

    # prove ok by above. so, 
    encoded_jpg = buf_io.getvalue()
    encoded_jpg_io = buf_io
    image = newimage
    # raise ValueError('Image format not JPEG')
    # raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      diff = obj.get('difficult')
      if diff is None:
        difficult = False
      else:
        difficult = bool(int(diff))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
#      print("whyyyyyyyyyy")
#      print(obj['name'])

      name = obj['name'].lower()
      if name not in label_map_dict:
        print('ERROR, label not found, ', name ,' for image:', img_path)
        continue

      classes.append(label_map_dict[name])
      truncated_str = obj.get('truncated')
      if truncated_str is None:
        truncated_value = 0
      else:
        truncated_value = int(truncated_str)
      truncated.append(truncated_value)

      pose_str = obj.get('pose')
      if pose_str is None:
        pose_str = "Unspecified"
      poses.append(pose_str.encode('utf8'))
      # poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example

# def main_(_):
#   if FLAGS.set not in SETS:
#     raise ValueError('set must be in : {}'.format(SETS))

#   data_dir = FLAGS.data_dir

#   output_file = os.path.join(FLAGS.output_dir, 'construction_' + FLAGS.set + '.record')
#   writer = tf.python_io.TFRecordWriter(output_file)

#   label_map_file = os.path.join(FLAGS.label_map_file)  
#   label_map_dict = label_map_util.get_label_map_dict(label_map_file)
  
#   FLAGS.imageset_path
#   examples_path = os.path.join(FLAGS.imageset_path,
#                                  'construction_' + FLAGS.set + '.txt')
  
#   annotations_dir = os.path.join(FLAGS.annotations_dir)
#   examples_list = dataset_util.read_examples_list(examples_path)
  
#   for idx, example in enumerate(examples_list):
#     if idx % 5 == 0:
#       print('On image %d of %d', idx, len(examples_list))
#     path = os.path.join(annotations_dir, example + '.xml')
# #    print(path)

#     with tf.gfile.GFile(path, 'r') as fid:
#       xml_str = fid.read()
#     xml = etree.fromstring(xml_str)
#     data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

#     tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
#                                     FLAGS.ignore_difficult_instances)
#     writer.write(tf_example.SerializeToString())

#   writer.close()

def build_tf_record(train_list ,path, label_map_dict):

  writer = tf.python_io.TFRecordWriter(path)
  for idx, example in enumerate(train_list):
    print('File, ', example)
    if idx % 100 == 0:
      print('On image %d of %d', idx, len(train_list))

    with tf.gfile.GFile(example, 'r') as fid:
      xml_str = fid.read()
    # print('xml,', xml_str)

    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                    FLAGS.ignore_difficult_instances)
    writer.write(tf_example.SerializeToString())
    # break

  writer.close()


def main(_):
  
  data_dir = FLAGS.data_dir
  annotations_dir = os.path.join(data_dir, 'Annotations')

  label_map_file = os.path.join(FLAGS.label_map_file)  
  label_map_dict = label_map_util.get_label_map_dict(label_map_file)

  examples_list = glob.glob(os.path.join(annotations_dir, '*.xml'))
  random.shuffle(examples_list)
  
  total = len(examples_list)

  percent = 0.8
  

  train_list = examples_list[0: int(percent*total)]
  val_list = examples_list[int(percent*total):]
  print('Training size: ', len(train_list))
  print('Validation size: ',len(val_list))


  build_tf_record(train_list, os.path.join(data_dir, 'train.record'), label_map_dict)
  build_tf_record(val_list, os.path.join(data_dir, 'val.record'), label_map_dict)
  







if __name__ == '__main__':
  tf.app.run()