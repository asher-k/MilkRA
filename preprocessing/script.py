import preprocess as pp
import os
import cv2


def run(datapath, csv_exptpath, img_exptpath, annotate):
    """
    Main script (processes multiple folders of images to generate CSV & (potentially) annotated files)
    """
    experiments = os.listdir(datapath)
    pp.update_directories(csv_exptpath, img_exptpath)

    for ex in experiments:
        current = datapath + "/" + ex
        files = os.listdir(current)
        images = []
        for img in files:
            img = cv2.imread(current + "/" + img, cv2.IMREAD_GRAYSCALE)
            images.append(img)

        # Now begin the preprocessing
        refls = [pp.pp_refl(images[len(images) - 1])] * (len(images))
        midpoint = pp.pp_midpoint(images[0], refls[0])  # midpoint should be found when time = 2s

        w = [pp.pp_width(i, r) for i, r in zip(images, refls)]
        h = [pp.pp_height(i, r) for i, r in zip(images, refls)]
        _indicies = [hi[1] for hi in h]  # need to refactor to separate index from height
        h = [hi[0] for hi in h]
        mid_h = [pp._height(i, midpoint, r) for i, r in zip(images, refls)]  # height @ the midpoint

        # And export for this set of images
        pp.to_csv(pp.FEATURES, csv_exptpath, ex + ".csv", files, refls, w, h, mid_h)  # then start exporting process
        if annotate:
            if not os.path.exists(img_exptpath + "/" + ex):
                os.makedirs(img_exptpath + "/" + ex)
            pp.annotate_images(images, img_exptpath + "/" + ex, files, refls, h, _indicies, mid_h,
                               [midpoint] * len(images))

        print("Finished " + ex)
    print("DONE")
