import os
import glob
import re
import shutil
import pandas as pd


def main():
    dataset_folder = 'safety'
    IMAGE_FOLDER = 'JPEGImages'
    BACKUP_IMAGE_FOLDER = '_' + IMAGE_FOLDER
    SEGMENTATION_FOLDER = dataset_folder + os.path.sep + 'SegmentationClass'

    image_source_folder = dataset_folder + os.path.sep + BACKUP_IMAGE_FOLDER
    image_dest_folder = dataset_folder + os.path.sep + IMAGE_FOLDER

    # prepare folder for new image set based on segmentation
    if not os.path.isdir(dataset_folder + os.path.sep + BACKUP_IMAGE_FOLDER):
        os.rename(dataset_folder + os.path.sep + IMAGE_FOLDER, dataset_folder + os.path.sep + BACKUP_IMAGE_FOLDER)
        os.mkdir(dataset_folder + os.path.sep + IMAGE_FOLDER)
    else:
        print('Found, ', dataset_folder + os.path.sep + BACKUP_IMAGE_FOLDER, ' ')

    segmentation_images = glob.glob(SEGMENTATION_FOLDER + os.path.sep + '*.png')

    for f in segmentation_images:
        # print(f)
        strs = f.split(os.path.sep)
        seg_filename = strs[len(strs) - 1]  # e.g. frame_000000.png
        index = re.split('_|\.', seg_filename)

        i = int(index[1])
        # print(index[1], ' : ',i)

        src_jpg = image_source_folder + os.path.sep + str(i) + '.jpg'
        dest_jpg = image_dest_folder + os.path.sep + 'frame_' + str(i).zfill(6) + '.jpg'
        shutil.copyfile(src_jpg, dest_jpg)

        # break

#    buildFilesList(dataset_folder, segmentation_images)

    print("Completed. ")


def buildFilesList(dataset_folder, segmentation_images):
    images_filenames = pd.DataFrame(segmentation_images)
    images_filenames = images_filenames[0].apply(lambda x: x.split(os.path.sep)[2])
    images_filenames = images_filenames.str.replace('.png', '')
    images_filenames = images_filenames.sample(frac=1)
    total = len(images_filenames)
    train_size = int(total * 0.8)
    train = images_filenames[:train_size]
    val = images_filenames[train_size:]
    trainval = images_filenames
    folder = os.path.join(dataset_folder, 'ImageSets', 'Segmentation')
    train.to_csv(os.path.join(folder, 'train.txt'), index=False)
    val.to_csv(os.path.join(folder, 'val.txt'), index=False)
    trainval.to_csv(os.path.join(folder, 'trainval.txt'), index=False)

    default_file = os.path.join(folder, 'default.txt')
    if os.path.isfile(default_file):
        os.remove(default_file)


if __name__ == '__main__':
    main()
