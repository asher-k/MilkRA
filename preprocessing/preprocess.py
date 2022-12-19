import pandas as pd
from Image import Droplet, CONSTS
from pair import PointPair
import cv2
import os


FEATURES = ["file", "reflection_row", "dl_reflection_width", "dl_height_midpoint"]  # Named .csv columns (no pairs!!)


def annotate_images(imgs, fpath, fnames, *data):
    """
    Draws annotations on images at the reference line and maximum height of the droplet
    :return:
    """
    _ = [*data]
    _ = _.pop()
    data = data[:-1]
    pair_data = []
    [pair_data.extend([p.l_values, p.r_values, [p.l_index] * len(p.l_values), [p.r_index] * len(p.r_values)]) for p in _]

    for im, name, d, p in zip(imgs, fnames, zip(*data), zip(*pair_data)):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)  # convert from gs to rgb
        ref_line = d[0]
        midpoint_heights = d[1]
        midpoint_locations = d[2]
        lefts = d[3]
        ref_w = d[4]

        for i in range(0, len(im[ref_line])):
            im[ref_line][i] = [60, 0, 164]  # highlight reference line in red

        for r in range(ref_line-CONSTS.REF_RADIUS*15, ref_line):  # should be total of ~60px high
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

            if l_val is not None:  # When the height values are None just skip
                for left in range(ref_line, ref_line - int(l_val), -1):
                    im[left][l_ind] = [255, 255, 0]
            if r_val is not None:  # When the height values are None just skip
                for right in range(ref_line, ref_line - int(r_val), -1):
                    im[right][r_ind] = [255, 255, 0]

        cv2.imwrite(f"{fpath}/{name}", im)


def to_csv(titles, fpath, fname, *data, point_mean=False):
    """
    Exports the calculated features of droplet image data to a CSV format.

    :return:
    """
    data = [*data]
    feats = data[0: 2]
    data = data[2:]
    # expands value of each PointPair Object in *data
    for d in data:
        if type(d[0]) is PointPair:
            [feats.extend([p.l_values, p.r_values]) if not point_mean else feats.append(p.pair_mean()) for p in d]
        else:
            feats.append(d)  # case for midpoint

    assert len(titles) == len(feats)
    feats = pd.DataFrame(list(zip(*feats)), columns=titles)
    feats.to_csv(f"{fpath}/{fname}")


def update_directories(csvpath, imgpath):
    """
    Creates export directories if they do not exist

    :param csvpath: export path of .csv files
    :param imgpath: export path of annotated images
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
    true_observations = int(desired_observations + 2) if padded else desired_observations  # if padded add edge areas
    interval_size = droplet_width//true_observations

    for i in range(0, desired_observations//2):  # Only take 1/2 observations since we're constructing PAIRs
        left = midpoint-(droplet_width//2) + ((true_observations-desired_observations)//2) * interval_size + (i*interval_size)
        right = midpoint+(droplet_width//2) - ((true_observations-desired_observations)//2) * interval_size - (i*interval_size)

        p = PointPair(rname=str(desired_observations//2-i)+"r", lname=str(desired_observations//2-i)+"l",
                      rind=right, lind=left)
        pairs.append(p)

    if padded:      # Then add 4 additional pairs at smaller intervals in the padded zone
        padded_interval_size = (((true_observations-desired_observations)//2) * 2 * interval_size) // 6
        for i in range(0, 4):
            buffer = 1 if i < 2 else 2
            left = midpoint - (droplet_width // 2) + (i+buffer) * padded_interval_size
            right = midpoint + (droplet_width // 2) - (i+buffer) * padded_interval_size
            p = PointPair(rname="edge_" + str(4-i) + "_l", lname="edge_" + str(4-i) + "_r", rind=right, lind=left)
            pairs.append(p)
    return pairs


def run(datapath, dataset, csv_exptpath, img_exptpath, annotate, height_method):
    """
    Main script (processes a single folder of images to generate CSV & (potentially) annotated files)
    """

    # Establish path environment
    combined_path = datapath + "/" + dataset
    files = os.listdir(combined_path)
    update_directories(csv_exptpath, img_exptpath)
    images = []
    for img in files:
        img = cv2.imread(combined_path + "/" + img, cv2.IMREAD_GRAYSCALE)
        images.append(img)

    print("Preprocessing", dataset + "...")
    _refl = Droplet(images[-1], height_method)._reflection()
    droplets = [Droplet(img, height_method, refl=_refl).setup() for img in images]

    unprocessed_fearures = FEATURES.copy()
    pairs = construct_pairs(droplets[-1].wid, 22, droplets[-1].mid, padded=True)
    for p in pairs:
        p.l_values = [drop.height_average(p.l_index, CONSTS.HEIGHT_RADIUS) for drop in droplets]
        p.r_values = [drop.height_average(p.r_index, CONSTS.HEIGHT_RADIUS) for drop in droplets]
        unprocessed_fearures.extend([p.l_name, p.r_name])
    to_csv(unprocessed_fearures,
           csv_exptpath,
           f"{dataset}_raw.csv",
           files,
           [_refl]*len(droplets),
           [d.wid for d in droplets],
           [d.hgts[d.mid] for d in droplets],
           pairs)

    # Features of processed data
    processed_features = [FEATURES[i] for i in [0, 2]]+[p.merged_title() for p in pairs[0:11]]+[FEATURES[3]] + \
                         [p.merged_title() for p in pairs[11:13]]+[pairs[0].merged_title()] + \
                         [p.merged_title() for p in pairs[13:]]  # manual override per Gideon
    to_csv(processed_features,
           csv_exptpath,
           f"{dataset}_processed.csv",
           files,
           [d.wid for d in droplets],
           pairs[0:11],
           [d.hgts[d.mid] for d in droplets],
           pairs[11:13],
           [pairs[0]],
           pairs[13:],
           point_mean=True)
    print("Exported csv files: ", csv_exptpath+"/"+dataset)

    if annotate:
        if not os.path.exists(img_exptpath + "/" + dataset):
            os.makedirs(img_exptpath + "/" + dataset)

        annotate_images(images,
                        f"{img_exptpath}/{dataset}",
                        files,
                        [_refl]*len(droplets),
                        [d.hgts[d.mid] for d in droplets],
                        [d.mid for d in droplets],
                        [d.l for d in droplets],
                        [d.wid for d in droplets],
                        pairs)
        print("Exported annotations: ", img_exptpath + "/" + dataset)
