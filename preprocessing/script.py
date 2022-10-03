import preprocess as pp
import os
import cv2
from progress.bar import IncrementalBar


def run(datapath, csv_exptpath, img_exptpath, annotate, height_method):
    """
    Main script (processes multiple folders of images to generate CSV & (potentially) annotated files)
    """
    # Define correct height methodology
    if height_method == "top":
        height = pp._height_top
    elif height_method == "bottom":
        height = pp._height
    else:
        raise Exception("Unknown height method, should be either \"top\" or \"bottom\"")

    # Establish path environment
    experiments = os.listdir(datapath)
    pp.update_directories(csv_exptpath, img_exptpath)
    with IncrementalBar('Processing Imagesets', max=len(experiments) * 6) as bar:
        for ex in experiments:
            current = datapath + "/" + ex
            files = os.listdir(current)
            images = []
            for img in files:
                img = cv2.imread(current + "/" + img, cv2.IMREAD_GRAYSCALE)
                images.append(img)

            # Now begin the preprocessing
            print(" Preprocessing", ex, "...")
            refls = [pp.pp_refl(images[len(images) - 1])] * (len(images))
            midpoint = pp.pp_midpoint(images[0], refls[0])  # midpoint should be found when time = 2s
            bar.next()

            # max_w = [pp_width(i, r) for i, r in zip(images, refls)]  # maximum width of the droplet
            ref_w = [pp._width(i[r]) for i, r in zip(images, refls)]
            bar.next()
            lefts = [pp.find_left(i, r) for i, r in zip(images, refls)]
            bar.next()
            mid_h = [height(i, midpoint, r, pp.HEIGHT_RADIUS) for i, r in zip(images, refls)]  # height @ the midpoint
            bar.next()

            # Heights at even intervals on each side of the midpoint
            pairs = pp.construct_pairs(min(ref_w), 22, midpoint, padded=True)
            for p in pairs:
                p.l_values = [height(im, p.l_index, r, pp.HEIGHT_RADIUS) for im, r in zip(images, refls)]
                p.r_values = [height(im, p.r_index, r, pp.HEIGHT_RADIUS) for im, r in zip(images, refls)]
                pp.FEATURES.extend([p.l_name, p.r_name])

            processed_features = [pp.FEATURES[0]] + [p.merged_title() for p in
                                                     pairs]  # Features for the PROCESSED (not raw) .csv
            pp.to_csv(pp.FEATURES, csv_exptpath, ex + "_raw.csv", files, refls, ref_w, mid_h, pairs)
            pp.to_csv(processed_features, csv_exptpath, ex + "_processed.csv", files, pairs, point_mean=True)
            print(" Exported csv files: ", csv_exptpath + "/" + ex)
            bar.next()

            if annotate:
                if not os.path.exists(img_exptpath + "/" + ex):
                    os.makedirs(img_exptpath + "/" + ex)
                pp.annotate_images(images, img_exptpath + "/" + ex, files, refls, mid_h, [midpoint] * len(images), lefts
                                   , ref_w, pairs)
                print(" Exported annotations: ", img_exptpath + "/" + ex)

            bar.next()
