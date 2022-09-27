import pandas as pd
import numpy as np
from pair import PointPair
import cv2
import os


# IMAGE variables
REF_RADIUS = 10  # Radius of the search area when attempting to find the reflection line of an image
REF_THRESH = 2.0  # Maximum difference between each side of the radius for a row to be considered reflected
REF_DROP_PXL = 50  # Maximum value of a BW pixel for it to be considered part of the droplet (when finding reflection)
REF_DROP_PXL_BORDER = 152  # Maximum value of a BW pixel for it to be considered the droplet (when finding height)
REF_NONDROP = 225  # Minimum value of a BW pixel for it to be considered not part of the droplet (when finding sides)
REF_LB = 700  # Lower Bound where pixels below are guaranteed to not be part of the Droplet (ie only reflection)
HEIGHT_RADIUS = 10  # Number of columns to each side of a given pixel to average over (enables smoother estimations)
FEATURES = ["file", "reflection_row", "dl_reflection_width", "dl_height_midpoint"]  # Named .csv columns (no pairs!!)


def find_left(img, r):
    """
    Finds the leftmost point of the droplet at the current row

    :return:
    """
    row = img[r]
    for index, r in enumerate(row):
        if r < REF_NONDROP:
            return index
    raise Exception("Unable to find left side of droplet")


def _height_top(image, col_index, ref, avg=5):
    """
    Calculates the height of a column of a droplet image (in pixels) by taking an average of a surrounding radius, by
    measuring from the top of the image down.

    :return:
    """
    heights = []
    for col_index in range (col_index-avg, col_index+avg+1, 1):
        height = 0
        while image[height][col_index] > REF_DROP_PXL_BORDER:
            height += 1
        height = ref-height
        heights.append(height)
    return int(np.mean(heights))


def _height(image, col_index, ref, avg=5):
    """
    Calculates the height of a column of a droplet image (in pixels) by taking an average of a surrounding radius, by
    measuring from the bottom of the droplet up.

    :return:
    """
    heights = []
    for col_index in range (col_index-avg, col_index+avg+1, 1):
        height = 0
        while image[ref-height][col_index] < REF_DROP_PXL_BORDER:
            height += 1
        heights.append(height)
    return int(np.mean(heights))


def _width(row):
    """
    Calculates the width of a row of a droplet image (in pixels)

    :return:
    """
    left_size = 0  # non-droplet pixels to the left of the droplet
    right_size = 0  # same as above, but to the right
    for index, r in enumerate(row):
        if r < REF_NONDROP and left_size == 0:  # start of the droplet
            left_size = index
            continue
        elif r > REF_NONDROP and left_size != 0 and right_size == 0:  # end of the droplet
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
    REF_LB = len(image)
    ref = REF_LB - REF_RADIUS
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
    :return:
    """
    _ = [*data]
    _ = _.pop()
    data = data[:-1]
    pair_data = []
    [pair_data.extend([p.l_values, p.r_values, [p.l_index] * len(p.l_values), [p.r_index] * len(p.r_values)])
     for p in _]

    for im, name, d, p in zip(imgs, fnames, zip(*data), zip(*pair_data)):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)  # convert from gs to rgb
        ref_line = d[0]
        midpoint_heights = d[1]
        midpoint_locations = d[2]
        lefts = d[3]
        ref_w = d[4]

        for i in range(0, len(im[ref_line])):
            im[ref_line][i] = [60, 0, 164]  # highlight reference line in red

        for r in range(ref_line-REF_RADIUS*15, ref_line):  # should be total of ~60px high
            im[r][lefts] = [253, 160, 2]  # highlight estimated ends of droplet in orange
            im[r][lefts+ref_w] = [253, 160, 2]

        for r in range(ref_line, ref_line-midpoint_heights, -1):
            im[r][midpoint_locations] = [0, 164, 60]  # highlight middle height in green

        # Now we start with the pairs
        for pi in range(0, len(p)//4):
            l_val = p[0 + (4*pi)]
            r_val = p[1 + (4*pi)]
            l_ind = p[2 + (4*pi)]
            r_ind = p[3 + (4*pi)]

            for left in range(ref_line, ref_line - l_val, -1):
                im[left][l_ind] = [255, 255, 0]
            for right in range(ref_line, ref_line - r_val, -1):
                im[right][r_ind] = [255, 255, 0]

        cv2.imwrite(fpath+"/"+name, im)


def to_csv(titles, fpath, fname, *data, point_mean=False):
    """
    Exports the calculated features of droplet image data to a CSV format.

    :return:
    """
    data = [*data]
    pair_data = data.pop()
    # Correctly expands value of each PointPair Object
    [data.extend([p.l_values, p.r_values]) if not point_mean else data.append(p.pair_mean()) for p in pair_data]

    assert len(titles) == len(data)
    data = pd.DataFrame(list(zip(*data)), columns=titles)
    data.to_csv(fpath+"/"+fname)


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


def construct_pairs(droplet_width, desired_observations, midpoint, padded=True):
    """
    Defines and updates all pair objects

    :param droplet_width: Width of the droplet (in pixels)
    :param desired_observations: Number of heights to draw in the image (/2 for number of pairs)
    :param midpoint: integer midpoint of the droplet
    :param padded: True if greater space should be given to edge values; also True if edge values are recorded.
    :return: 
    """
    assert desired_observations % 2 == 0

    pairs = []
    true_observations = int(desired_observations * 1.2) if padded else desired_observations  # if padded add edge areas
    interval_size = droplet_width//true_observations

    for i in range(0, desired_observations//2):  # Only take 1/2 observations since we're constructing PAIRs
        left = midpoint-(droplet_width//2) + ((true_observations-desired_observations)//2) * interval_size + (i*interval_size)
        right = midpoint+(droplet_width//2) - ((true_observations-desired_observations)//2) * interval_size - (i*interval_size)

        p = PointPair(rname=str(desired_observations//2-i)+"r", lname=str(desired_observations//2-i)+"l",
                      rind=right, lind=left)
        pairs.append(p)

    if padded:      # Then add 5 additional pairs at smaller intervals in the padded zone
        padded_interval_size = (((true_observations-desired_observations)//2) * interval_size) // 7
        for i in range(0, 5):
            left = midpoint - (droplet_width // 2) + (i+1) * padded_interval_size
            right = midpoint + (droplet_width // 2) - (i+1) * padded_interval_size
            p = PointPair(rname="edge_" + str(5-i) + "_l", lname="edge_" + str(5-i) + "_r", rind=right, lind=left)
            pairs.append(p)
    return pairs


def run(datapath, dataset, csv_exptpath, img_exptpath, annotate, height_method):
    """
    Main script (processes a single folder of images to generate CSV & (potentially) annotated files)
    """
    # Define correct height methodology
    if height_method == "top":
        height = _height_top
    elif height_method == "bottom":
        height = _height
    else:
        raise Exception("Unknown height method, should be either \"top\" or \"bottom\"")

    # Establish path environment
    combined_path = datapath + "/" + dataset
    files = os.listdir(combined_path)
    update_directories(csv_exptpath, img_exptpath)
    images = []
    for img in files:
        img = cv2.imread(combined_path + "/" + img, cv2.IMREAD_GRAYSCALE)
        images.append(img)

    print("Preprocessing", dataset, "...")
    refls = [pp_refl(images[len(images) - 1])] * (len(images))  # assumes static reflection across all images of set
    midpoint = pp_midpoint(images[0], refls[0])  # midpoint should be found when time = 2s

    ref_w = [_width(i[r]) for i, r in zip(images, refls)]  # width of the droplet at the found reflection line
    lefts = [find_left(i, r) for i, r in zip(images, refls)]  # left side of the droplet (needed to display widths)
    mid_h = [height(i, midpoint, r, HEIGHT_RADIUS) for i, r in zip(images, refls)]  # height @ the midpoint

    # Heights at even intervals on each side of the midpoint
    pairs = construct_pairs(ref_w[0], 20, midpoint, padded=True)
    for p in pairs:
        p.l_values = [height(im, p.l_index, r, HEIGHT_RADIUS) for im, r in zip(images, refls)]
        p.r_values = [height(im, p.r_index, r, HEIGHT_RADIUS) for im, r in zip(images, refls)]
        FEATURES.extend([p.l_name, p.r_name])

    processed_features = [FEATURES[0]] + [p.merged_title() for p in pairs]  # Features for the PROCESSED (not raw) .csv
    to_csv(FEATURES, csv_exptpath, dataset+"_raw.csv", files, refls, ref_w, mid_h, pairs)
    to_csv(processed_features, csv_exptpath, dataset+"_processed.csv", files, pairs, point_mean=True)
    print("Exported csv files: ", csv_exptpath+"/"+dataset)

    if annotate:
        if not os.path.exists(img_exptpath + "/" + dataset):
            os.makedirs(img_exptpath + "/" + dataset)

        annotate_images(images, img_exptpath + "/" + dataset, files, refls, mid_h, [midpoint] * len(images), lefts,
                        ref_w, pairs)
        print("Exported annotations: ", img_exptpath + "/" + dataset)
