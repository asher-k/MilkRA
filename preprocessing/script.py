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
    with IncrementalBar('Processing Imagesets', max=len(experiments) * 7) as bar:
        for ex in experiments:
            print("")  # otherwise next print overlaps
            current = datapath + "/" + ex
            files = os.listdir(current)
            images = []
            for img in files:
                img = cv2.imread(current + "/" + img, cv2.IMREAD_GRAYSCALE)
                images.append(img)

            # Now begin the preprocessing
            print("Preprocessing", ex, "...")
            refls = [pp.pp_refl(images[len(images) - 1])] * (len(images))
            midpoint = pp.pp_midpoint(images[0], refls[0])  # midpoint should be found when time = 2s
            bar.next()

            # max_w = [pp_width(i, r) for i, r in zip(images, refls)]  # maximum width of the droplet
            ref_w = [pp._width(i[r]) for i, r in zip(images, refls)]
            bar.next()
            lefts = [pp.find_left(i[r]) for i, r in zip(images, refls)]
            bar.next()

            h = [pp.pp_height(i, r) for i, r in zip(images, refls)]
            _indicies = [hi[1] for hi in h]  # need to refactor to separate index from height
            h = [hi[0] for hi in h]
            bar.next()

            mid_h = [height(i, midpoint, r, pp.HEIGHT_RADIUS) for i, r in zip(images, refls)]  # height @ the midpoint
            bar.next()

            # Heights at even intervals on each side of the midpoint
            interval_size = ref_w[0] // 12
            interval_heights = []
            for i in range(5, 0, -1):  # before the midpoint
                interval = [height(im, midpoint - (interval_size * i), r, pp.HEIGHT_RADIUS) for im, r in zip(images, refls)]
                interval_heights.append(interval)
            for i in range(1, 6):  # after the midpoint
                interval = [height(im, midpoint + (interval_size * i), r, pp.HEIGHT_RADIUS) for im, r in zip(images, refls)]
                interval_heights.append(interval)
            bar.next()

            # And export for this set of images
            pp.to_csv(pp.FEATURES, csv_exptpath, ex + ".csv", files, refls, ref_w, mid_h, interval_heights[0],
                      interval_heights[1], interval_heights[2], interval_heights[3], interval_heights[4],
                      interval_heights[5], interval_heights[6], interval_heights[7], interval_heights[8],
                      interval_heights[9])
            print("\nExported csv file: ", csv_exptpath + "/" + ex + ".csv")

            if annotate:
                if not os.path.exists(img_exptpath + "/" + ex):
                    os.makedirs(img_exptpath + "/" + ex)
                pp.annotate_images(images, img_exptpath + "/" + ex, files, refls, mid_h, [midpoint] * len(images), lefts
                                   , ref_w, [interval_size] * len(images), interval_heights[0], interval_heights[1],
                                   interval_heights[2], interval_heights[3], interval_heights[4], interval_heights[5],
                                   interval_heights[6], interval_heights[7], interval_heights[8], interval_heights[9])
                print("Exported annotations: ", img_exptpath + "/" + ex)

            bar.next()
