#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -ex


export CUDA_VISIBLE_DEVICES="0"
# export TF_ENABLE_AUTO_MIXED_PRECISION=1
# export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=1
# export TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING=1

# Train 10 iterations.
NUM_ITERATIONS=1000


# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
# good for environment test
# python "${WORK_DIR}"/model_test.py -v



# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
#sh download_and_convert_voc2012.sh
#sh prepare_safety.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
#PASCAL_FOLDER="pascal_voc_seg"
PASCAL_FOLDER="safety"

EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
# TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_ROOT="https://home.leeseng.tech/Projects/BuildingLab/deeplab/init_models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
cd "${INIT_FOLDER}"
# wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"


# remove training
rm -fr /home/leeseng/Projects/iss/BuildingLab/tensorflow/models/research/deeplab/datasets/safety/exp/train_on_trainval_set/train/*

# Train 10 iterations.
# NUM_ITERATIONS=10000

python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="trainval" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513,513 \
  --train_batch_size=2 \
  --dataset="safety" \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --initialize_last_layer=false \
  --last_layers_contain_logits_only=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --label_weights=1 \
  --label_weights=1 \
  --label_weights=1 \
  --label_weights=50 \
  --label_weights=50 \
  --label_weights=40 \
  --label_weights=10 \
  --label_weights=50 \
  --label_weights=100

# simple safety
#  --label_weights=1 \
#  --label_weights=1000 \
#  --label_weights=400 \
#  --label_weights=200



# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=1080,1920 \
  --dataset="safety" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --max_number_of_evaluations=1

# Visualize the results.
python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=1081,1921 \
  --dataset="safety" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --max_number_of_iterations=1


# Export the trained checkpoint.

#CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
#EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"
#
#python "${WORK_DIR}"/export_model.py \
#  --logtostderr \
#  --checkpoint_path="${CKPT_PATH}" \
#  --export_path="${EXPORT_PATH}" \
#  --model_variant="xception_65" \
#  --atrous_rates=6 \
#  --atrous_rates=12 \
#  --atrous_rates=18 \
#  --output_stride=16 \
#  --decoder_output_stride=4 \
#  --num_classes=4 \
#  --crop_size=513 \
#  --crop_size=513 \
#  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
