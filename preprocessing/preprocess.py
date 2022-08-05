import pandas as pd
import numpy as np
import cv2
import os


# IMAGE variables
REF_RADIUS = 10  # Radius of the search area when attempting to find the reflection line of an image
REF_THRESH = 2.0  # Maximum difference between each side of the radius for a row to be considered reflected
REF_DROP_PXL = 50  # Maximum value of a BW pixel for it to be considered part of the droplet (when finding reflection)
REF_DROP_PXL_BORDER = 128  # Maximum value of a BW pixel for it to be considered the droplet (when finding height)
REF_LB = 700  # Lower Bound where pixels below are guaranteed to not be part of the Droplet (ie only reflection)
FEATURES = ["file", "reference_row", "dl_width", "dl_height_abs", "dl_midpoint_height"]  # Named columns in the .csv


def _height(image, col_index, ref):
    """
    Calculates the height of a column of a droplet image (in pixels)

    :return:
    """
    height = 0
    while image[ref-height][col_index] < REF_DROP_PXL_BORDER:
        height += 1
    return height


def _width(row):
    """
    Calculates the width of a row of a droplet image (in pixels)

    :return:
    """
    left_size = 0  # non-droplet pixels to the left of the droplet
    right_size = 0  # same as above, but to the right
    for index, r in enumerate(row):
        if r < REF_DROP_PXL and left_size == 0:  # start of the droplet
            left_size = index
            continue
        elif r > REF_DROP_PXL and left_size != 0 and right_size == 0:  # end of the droplet
            right_size = len(row)-index
            break

    if left_size == 0 and right_size == 0:  # case for when all white pixels (ie no droplet)
        return 0
    return len(row)-(left_size+right_size)


def pp_height(image, ref=None):
    """
    Calculates the height of the supplied droplet images.
    The height is defined as the maximum pixel distance between the top of the droplet and the reference line.

    :return:
    """
    if ref is None:
        ref == pp_refl(image)

    heights = []
    for col in range(0, len(image[0])):
        heights.append(_height(image, col, ref))

    return np.max(heights), np.where(heights == np.max(heights))


def pp_width(image, ref=None):
    """
    Calculates the maximum width of the supplied droplet image.
    The width is defined as the maximum pixel distance between both sides of the droplet above the reference line.
    Note the width is not always guaranteed to be at the base of the droplet and fluctuates over its lifespan.

    :return:
    """
    if ref is None:
        ref == pp_refl(image)

    width = []
    for row in image[0:REF_LB]:
        w = _width(row)
        width.append(w)

    return np.max(width)


def pp_refl(image):
    """
    Calculates the reference line for the droplet image.
    The 'reference line' is defined as where the base of the droplet meets its reflection.
    :param image:
    :return: a list of numbers representing the rows of reference within the images
    """
    ref = REF_LB - REF_RADIUS  # TODO: check if r or c
    while ref > REF_RADIUS:  # for each row, check widths of lines above & below the current
        pre = [_width(i) for i in image[ref-REF_RADIUS:ref]]
        post = [_width(i) for i in image[ref:ref+REF_RADIUS]]
        # print("Found at ", ref, ": ", np.mean(np.subtract(pre, post)))
        if np.abs(np.mean(np.subtract(pre, post))) <= REF_THRESH:  # if within threshold, we have our reflection!
            return ref
        ref -= 1
    raise Exception("Unable to find reflection line")


def pp_midpoint(image, ref):
    """
    Calculates the midpoint on the droplet within the image.

    :return:
    """
    mp = 0
    left = 0
    while mp < len(image[ref]):
        if image[ref][mp] >= REF_DROP_PXL >= image[ref][mp + 1]:  # found left side of droplet
            left = mp
        elif image[ref][mp] <= REF_DROP_PXL <= image[ref][mp + 1]:  # found right side of droplet
            return (mp-left)//2 + left
        mp += 1
    return mp


def annotate_images(imgs, fpath, fnames, *data):
    """
    Draws annotations on images at the reference line and maximum height of the droplet
    *data should be in order of REF, HEIGHT, HGHT_IND, MID_HGHT, MIDPOINTs

    :return:
    """
    for im, name, d in zip(imgs, fnames, zip(*data)):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)  # convert from gs to rgb

        for i in range(0, len(im[d[0]])):
            im[d[0]][i] = [60, 0, 164]  # highlight reference line in red

        for r in range(d[0], d[0]-d[1], -1):
            im[r][d[2]] = [164, 0, 60]  # highlight maximum height in blue

        for r in range(d[0], d[0]-d[3], -1):
            im[r][d[4]] = [0, 164, 60]  # highlight middle height in blue

        cv2.imwrite(fpath+"/"+name, im)

    print("ANNOTATION SUCCESSFUL")


def to_csv(titles, fpath, fname, *data):
    """
    Exports the calculated features of droplet image data to a CSV format.

    :return:
    """
    data = [*data]
    assert len(titles) == len(data)
    data = pd.DataFrame(list(zip(*data)), columns=titles)
    data.to_csv(fpath+"/"+fname)
    print("EXPORT SUCCESSFUL")


def update_directories(csvpath, imgpath):
    """
    Creates export directories if they do not exist

    :param csvpath:
    :param imgpath:
    :return:
    """
    if not os.path.exists(csvpath):
        os.makedirs(csvpath)
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)


def run(datapath, dataset, csv_exptpath, img_exptpath, filename, annotate):
    """
    Main script (processes a single folder of images to generate CSV & (potentially) annotated files)
    """
    combined_path = datapath + "/" + dataset
    files = os.listdir(combined_path)
    update_directories(csv_exptpath, img_exptpath)

    images = []
    for img in files:
        img = cv2.imread(combined_path + "/" + img, cv2.IMREAD_GRAYSCALE)
        images.append(img)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)

    refls = [pp_refl(images[len(images) - 1])] * (len(images))  # assumes static reflection across all images of set
    midpoint = pp_midpoint(images[0], refls[0])  # midpoint should be found when time = 2s

    w = [pp_width(i, r) for i, r in zip(images, refls)]
    h = [pp_height(i, r) for i, r in zip(images, refls)]
    _indicies = [hi[1] for hi in h]  # need to refactor to separate index from height
    h = [hi[0] for hi in h]

    mid_h = [_height(i, midpoint, r) for i, r in zip(images, refls)]  # height @ the midpoint

    to_csv(FEATURES, csv_exptpath, filename, files, refls, w, h, mid_h)  # then start exporting process
    print("Exported .csv file %s", filename)

    if annotate:
        if not os.path.exists(img_exptpath + "/" + dataset):
            os.makedirs(img_exptpath + "/" + dataset)
        annotate_images(images, img_exptpath + "/" + dataset, files, refls, h, _indicies, mid_h, [midpoint] * len(images))
        print("Exported annotations")

    print("DONE")
