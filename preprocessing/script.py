import preprocess as pp
import os
import cv2

# Directory Settings
DATAPATH = "../data/raw"
CSV_EXPTPATH = "../data/o_features"
IMG_EXPTPATH = "../data/o_annotations"

# Export Settings
ANNOTATE = True
FEATURES = ["file", "reference_row", "dl_width", "dl_height_abs", "dl_midpoint_height"]

# main script (processes multiple folders of images to generate CSV & (potentially) annotated files
if __name__ == "__main__":
    experiments = os.listdir(DATAPATH)
    for ex in experiments:
        CURRENT = DATAPATH + "/" + ex
        files = os.listdir(CURRENT)
        images = []
        for img in files:
            img = cv2.imread(CURRENT + "/" + img, cv2.IMREAD_GRAYSCALE)
            images.append(img)

        # Now begin the preprocessing
        refls = [pp.pp_refl(images[len(images) - 1])] * (len(images))
        midpoint = pp.pp_midpoint(images[0], refls[0])  # midpoint should be found when time = 2s

        w = [pp.pp_width(i, r) for i, r in zip(images, refls)];
        h = [pp.pp_height(i, r) for i, r in zip(images, refls)];
        _indicies = [hi[1] for hi in h]  # need to refactor to separate index from height
        h = [hi[0] for hi in h]
        mid_h = [pp._height(i, midpoint, r) for i, r in zip(images, refls)]  # height @ the midpoint

        # And export for this set of images
        pp.to_csv(FEATURES, CSV_EXPTPATH, ex+".csv", files, refls, w, h, mid_h)  # then start exporting process
        if ANNOTATE:
            if not os.path.exists(IMG_EXPTPATH+"/"+ex):
                os.makedirs(IMG_EXPTPATH+"/"+ex)
            pp.annotate_images(images, IMG_EXPTPATH+"/"+ex, files, refls, h, _indicies, mid_h, [midpoint] * len(images))

        print("Finished " + ex)
