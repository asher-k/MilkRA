import preprocess as pp
import os
import cv2
from progress.bar import IncrementalBar


def run(datapath, csv_exptpath, img_exptpath, annotate):
    """
    Main script (processes multiple folders of images to generate CSV & (potentially) annotated files)
    """
    experiments = os.listdir(datapath)
    pp.update_directories(csv_exptpath, img_exptpath)
    with IncrementalBar('Processing Imagesets', max=len(experiments)*5) as bar:
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

            mid_h = [pp._height(i, midpoint, r) for i, r in zip(images, refls)]  # height @ the midpoint
            bar.next()

            # And export for this set of images
            pp.to_csv(pp.FEATURES, csv_exptpath, ex + ".csv", files, refls, ref_w, h, mid_h)  # then start exporting process
            print("\nExported csv file: ", csv_exptpath + "/" + ex + ".csv")

            if annotate:
                if not os.path.exists(img_exptpath + "/" + ex):
                    os.makedirs(img_exptpath + "/" + ex)
                pp.annotate_images(images, img_exptpath + "/" + ex, files, refls, h, _indicies, mid_h,
                                   [midpoint] * len(images), lefts, ref_w)
                print("Exported annotations: ", img_exptpath + "/" + ex)

            bar.next()
