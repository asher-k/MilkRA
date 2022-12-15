# File contains functions when cropping is enabled in main.py\
import os
import cv2
import preprocess as pp
import numpy as np
from Image import Droplet

BW_CROP = 230


def _find_crop(img):
    """
    Finds the crop point of an image (where the black value along one side rises above a threshold0
    """
    drop = Droplet(img, "top")
    img = drop.img
    vertical = None
    for i, row in enumerate(img):  # First find the vertical crop point
        if row[0] < BW_CROP and img[i+50][0] < BW_CROP and img[i+100][0] < BW_CROP and vertical is None:
            vertical = i
            break

    horizontal = drop._bounds(vertical)[0] - 100

    #  Then return the crop point
    return vertical if vertical is not None else len(img), horizontal


def _crop_at(img, row, col):
    """
    Crops the image at a given row/column pair
    """
    img = img[0:row]
    return np.array([i[col:] for i in img])


def crop_all(datapath, dataset, single=True, crop_dir=None):
    """
    Crops and exports every image of the defined imageset
    """
    if not os.path.exists(crop_dir):  # make directory if it does not exist
        os.makedirs(crop_dir)

    if single and dataset not in os.listdir(crop_dir):  # Only load if we have yet to crop the image
        print('Cropping images in', datapath + "/" + dataset)
        point = None
        for fname in os.listdir(datapath + "/" + dataset):
            img = cv2.imread(datapath + "/" + dataset + "/" + fname, cv2.IMREAD_GRAYSCALE)
            if point is None:
                point = _find_crop(img)
            img = _crop_at(img, *point)

            if not os.path.exists(crop_dir + "/" + dataset):
                os.makedirs(crop_dir + "/" + dataset)
            cv2.imwrite(crop_dir + "/" + dataset + "/" + fname, img)

    elif not single:  # We here iterate over each dataset and check if it's already been cropped
        experiments = os.listdir(datapath)
        for ex in experiments:
            if ex not in os.listdir(crop_dir):  # Only load if we have yet to crop the image
                print('Cropping images in', datapath + "/" + ex)
                point = None
                for fname in os.listdir(datapath + "/" + ex):
                    img = cv2.imread(datapath + "/" + ex + "/" + fname, cv2.IMREAD_GRAYSCALE)
                    if point is None:
                        point = _find_crop(img)
                    img = _crop_at(img, *point)

                    if not os.path.exists(crop_dir + "/" + ex):
                        os.makedirs(crop_dir + "/" + ex)
                    cv2.imwrite(crop_dir + "/" + ex + "/" + fname, img)
            else:
                print('Found cropped images in', crop_dir + "/" + ex)
    else:
        print('Found cropped images in', crop_dir + "/" + dataset)  # print for existing crops in single-image case

