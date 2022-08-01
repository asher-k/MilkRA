import pandas as pd
import numpy as np
import cv2
import os


def _height(image, col_index, ref):
    """
    Calculates the height of a column of a droplet image (in pixels)

    :return:
    """
    height = 0
    while image[ref-height][col_index] < REF_DROP_PXL:
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
        ref == pp_ref(image)

    heights = []
    for col in range(0, len(image[0])):
        heights.append(_height(image, col, ref))

    return np.max(heights)


def pp_width(image, ref=None):
    """
    Calculates the maximum width of the supplied droplet image.
    The width is defined as the maximum pixel distance between both sides of the droplet above the reference line.
    Note the width is not always guaranteed to be at the base of the droplet and fluctuates over its lifespan.

    :return:
    """
    if ref is None:
        ref == pp_ref(image)

    width = []
    for row in image[0:REF_LB]:
        width.append(_width(row))

    return np.max(width)


def pp_ref(image):
    """
    Calculates the reference line for the droplet image.
    The 'reference line' is defined as where the base of the droplet meets its reflection.
    :param image:
    :return: a list of numbers representing the rows of reference within the images
    """
    ref = REF_LB - REF_RADIUS  # TODO: check if r or c
    reflection = None
    while ref > REF_RADIUS:  # for each row, check widths of lines above & below the current
        pre = [_width(i) for i in image[ref-REF_RADIUS:ref]]
        post = [_width(i) for i in image[ref:ref+REF_RADIUS]]
        if np.abs(np.mean(np.subtract(pre, post))) <= REF_THRESH:  # if within threshold, we have our reflection!
            return ref
        ref -= 1
    raise Exception("Unable to find reflection line")


def to_csv(titles, *data):
    """
    Exports the calculated features of droplet image data to a CSV format.

    :param titles:
    :param data:
    :return:
    """
    data = [*data]
    assert len(titles) == len(data)
    data = pd.DataFrame(list(zip(*data)), columns=titles)
    data.to_csv(EXPTPATH+"/"+EXPTNAME)
    print("EXPORT SUCCESSFUL")


# PATH variables
DATAPATH = "../data/i_sidecam"
EXPTPATH = "../data/o_features"
EXPTNAME = "data.csv"

# IMAGE variables
REF_RADIUS = 10  # Radius of the search area when attempting to find the reflection line of an image
REF_THRESH = 0.5  # Maximum difference between each side of the radius for a row to be considered reflected
REF_DROP_PXL = 50  # Maximum value of a BW pixel for it to be considered part of the droplet
REF_LB = 700  # Lower Bound where pixels below are guaranteed to not be part of the Droplet (ie only reflection)

# main script
if __name__ == "__main__":
    files = os.listdir(DATAPATH)
    images = []
    for img in files:
        img = cv2.imread(DATAPATH+"/"+img, cv2.IMREAD_GRAYSCALE)
        images.append(img)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)

    # images = images[0:25]
    features = ["file", "reference_row", "dl_width", "dl_height"]

    refs = [pp_ref(images[49])]*50
    # refs = [pp_ref(i) for i in images]
    w = [pp_width(i, r) for i, r in zip(images, refs)]; print("done w")
    h = [pp_height(i, r) for i, r in zip(images, refs)]; print("done h")
    to_csv(features, files, refs, w, h)
