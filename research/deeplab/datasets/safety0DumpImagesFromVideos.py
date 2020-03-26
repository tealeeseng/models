import cv2
import glob
import os

HOME_FOLDER = '/home/leeseng/Projects/iss/BuildingLab/'
VIDEO_FOLDER = HOME_FOLDER + 'drive/data/hookcam/CCTV2/'
PROJECT_ROOT = HOME_FOLDER + 'tensorflow/models/research/deeplab/'
IMAGES_FOLDER = PROJECT_ROOT + 'datasets/safety_seg/VOCdevkit/VOC2012/JPEGImages/'


def main():
    files = glob.glob(VIDEO_FOLDER + '/*/*.mp4', recursive=True)
    f = files[0]
    for f in files:
       dumpImages(f)
#    print(len(files))

    cv2.destroyAllWindows()

    # open the file, take each 30 seconds image, dump to IMAGES_FOLDER


def dumpImages(f):
    filenames = f.split(os.path.sep)
    filename = filenames[len(filenames) - 1]


    cap = cv2.VideoCapture(f)
    fps = cap.get(cv2.CAP_PROP_FPS)
    FRAME_30 = fps * 30
    frameCount = 0
    dumpCount = 0

    if cap.isOpened() == False:
        print(f, ' is NOT opened.')

    print('Dumping: ', filename)

    while cap.isOpened():
        ret, frame = cap.read()
        frameCount = frameCount + 1

        if ret == True:
            if frameCount % FRAME_30 == 1:
                image_filename = IMAGES_FOLDER + filename + '-%d.jpg' % frameCount

                if os.path.isfile(image_filename):
                    print('Found images. Skipped ', filename)
                    break

                cv2.imwrite(image_filename, frame)
                # print('Dumped: ', image_filename)
                # cv2.imshow('Frame', frame)
                dumpCount = dumpCount+1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    if dumpCount !=0:
        print('Dumped: ', filename, ' images: ', dumpCount)


def main1():
    print(cv2.__version__)

if __name__ == '__main__':
    main()
