import pandas as pd
import numpy as np
import cv2
import os

DATAPATH = "../data/sidecam"


def pp_width(data, references=None):
    """
    Calculates the maximum width of all supplied droplet images.
    The width is defined as the maximum pixel distance between both sides of the droplet above the reference line.
    Note the width is not always guaranteed to be at the base of the droplet and fluctuates over its lifespan.

    :param data:
    :param references:
    :return:
    """
    if references is None:
        references == pp_ref(data)

    widths = []
    for i in data:
        pass

    return widths


def pp_height(data, references=None):
    """
    Calculates the height of all supplied droplet images.
    The height is defined as the maximum pixel distance between the top of the droplet and the reference line.

    :return:
    """
    if references is None:
        references == pp_ref(data)

    heights = []
    for i in data:
        pass

    return heights


def pp_ref(data):
    """
    Calculates the reference line for each supplied droplet image.
    The 'reference line' is defined as where the base of the droplet meets its reflection.
    :param data:
    :return: a list of numbers representing the rows of reference within the images
    """
    refs = []
    for i in data:
        pass

    return refs


def to_csv(titles, *data):
    """
    Exports the calculated features of droplet image data to a CSV format.

    :param titles:
    :param data:
    :return:
    """
    print(*data)


# main script
if __name__ == "__main__":
    files = os.listdir(DATAPATH)
    images = []
    for img in files:
        img = cv2.imread(DATAPATH+"/"+img)
        images.append(img)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)

    features = ["file", "dl_width", "dl_height"]
    w = pp_width(images)
    h = pp_height(images)
    to_csv(features, w, h, files)
