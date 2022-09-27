# File contains functions when cropping is enabled in main.py\
import os
import cv2


def _find_crop(img):
    """
    Finds the crop point of an image (where the black value along one side rises above a threshold0
    """
    crop_point = None
    for i, row in enumerate(img):  # First find the crop point
        if row[0] < 248 and img[i+50][0] < 248 and img[i+100][0] < 248 and crop_point is None:
            crop_point = i
            break

    #  Then return the crop point
    return crop_point if crop_point is not None else len(img)


def _crop_at(img, at):
    """
    Crops the image at a given row
    """
    return img[0:at]


def crop_all(datapath, dataset, single=True, crop_dir=None):
    """
    Crops and exports every image of the defined imageset
    """
    if not os.path.exists(crop_dir):  # make directory if it does not exist
        print('Cropping images in', datapath)
        os.makedirs(crop_dir)
    else:
        print('Found cropped images in', crop_dir)

    if single and dataset not in os.listdir(crop_dir):  # Only load if we have yet to crop the image
        point = None
        for fname in os.listdir(datapath + "/" + dataset):
            img = cv2.imread(datapath + "/" + dataset + "/" + fname, cv2.IMREAD_GRAYSCALE)
            if point is None:
                point = _find_crop(img)
            img = _crop_at(img, point)

            if not os.path.exists(crop_dir + "/" + dataset):
                os.makedirs(crop_dir + "/" + dataset)
            cv2.imwrite(crop_dir + "/" + dataset + "/" + fname, img)

    elif not single:  # We here iterate over each dataset and check if it's already been cropped
        experiments = os.listdir(datapath)
        for ex in experiments:
            if ex not in os.listdir(crop_dir):  # Only load if we have yet to crop the image
                point = None
                for fname in os.listdir(datapath + "/" + ex):
                    img = cv2.imread(datapath + "/" + ex + "/" + fname, cv2.IMREAD_GRAYSCALE)
                    if point is None:
                        point = _find_crop(img)
                    img = _crop_at(img, point)

                    if not os.path.exists(crop_dir + "/" + ex):
                        os.makedirs(crop_dir + "/" + ex)
                    cv2.imwrite(crop_dir + "/" + ex + "/" + fname, img)
