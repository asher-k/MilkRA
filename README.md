# Milk Droplet Analysis

## Preprocessing How-To

There are several assumptions made about the structure of the project which cannot be violated. You will first need to place your data into the `data/` directory. Each sub-directory within `data/` should contain a single set of droplet images. Please keep your naming conventions consistent, as folder and file names are used during the export process and can cause confusion if named ambiguously.

Please note that the input directory `data/` and the output directory `output/` are not cleared between runs; this is to ensure data is not accidentally lost. However, files within the `output/` directory are at risk of being overwritten when the program is run again. Copy results from the `output/` directory to your local device immediately after running the script if you wish to keep them.

### Running the Script

First activate your conda environment using the terminal command `conda activate milkra`. After this, navigate to the directory of the script, `cd path/to/MilkRA/preprocessing`. To run the script, use the terminal command `python main.py`. There are several command-line arguments available:

| Argument           | Value               | Description                                                                                                                                                                                         |
|--------------------------|--------------------------|--------------------|
| `--mode`           | `single` or `multi` | Mode of the script; under `single` only one provided sub-directory in `data/` is preprocessed. Under `multi` all sub-directories are preprocessed. Default `single`                                 |
| `--annotate`       |                     | Whether to export annotated versions of droplet images with the reflection line and midpoint height drawn.                                                                           |
| `--crop`           |                     | Whether to automatically crop droplet images vertically and save them for future use. If images have already been cropped in a previous execution, their cropped versions are used. |
| `--datapath`       | `<directory>`       | Relative path to the input directory. Default `../data`                                                                                                                                             |
| `--csv_exprpath`   | `<directory>`       | Relative path to the output directory for .csv files. Default `../output/csv`                                                                                                                       |
| `--img_exprpath`   | `<directory>`       | Relative path to the output directory for annotated image files. Default `../output/annotations`                                                                                                    |
| `--croppath`       | `<directory>`       | Relative path where cropped images are saved if cropping is performed. Default `../cropped`                                                                                                         |
| `--dataset`        | `<folder name>`     | **Required when** `--mode = single`. The subdirectory of `data/` to preprocess. Default `None`                                                                                                      |
| `--height_radius`  | `<integer>`         | Radius to use when calculating height, enables smoother height estimations. When `0`, only the target column's height is calculated (no smoothing). Default `2`                                    |
| `--height_method`  | `top` or `bottom`   | Method to calculate the height of the droplet, using either a top-down or bottom-up algorithm. Default `top` |
| `--width_only`     |                     | A special script mode which exports only the width and midpoint height values at each timestep. **Incompatible with** `--annotate`; please verify reflection points manually.|


The most important argument is `--mode`, which **must be added as an argument to any script run**. Note that in the `single` mode, the `--dataset` argument **must also be provided**. To add arguments to a script run, modify the terminal command used to run the python file: `python main.py --mode single --dataset example --annotate`.

### Recommendations
To ensure height measurements are correct it is recommended to run the script with `--annotate` and verify the annotated image files manually.  

In the ideal scenario (no measurement issues which would require a change of parameters & a re-run) an imageset only needs to be run once to obtain the .csv files. However, with `--annotate` and `--crop` the imageset is effectively copied twice in local storage. Therefore after an imageset has been processed without error it is strongly recommended that the `cropped/` and `output/annotations` directories are cleared.

### Troubleshooting
Occassionally scripts may produce errors - this is normal and is often the result of artifact(s) to the side of the droplet. Re-running these troublesome imagesets with `--crop` frequently fixes this issue. Should issues continue to persist, get in touch with Asher. Attach the imageset and describe your script settings.


### Understanding Extracted Features
In every run two outputs are always guaranteed in the form of `csv` files. `<name>_raw.csv` contains raw measurements for each point measured, along with other information about the imageset such as the location of the midpoint and reflection lines. `<name>_preprocessed.csv` only contains pairwise means for each measured pair. For a clearer illustration, see the figure below

![alt text](https://github.com/veryeager/milkra/blob/main/content/README_diagram.png?raw=true)

## Machine Learning
We make approach classification from several perspectives, including dimensionality reduction, time-series and computer vision methods. We define a set of baseline models including MLP, SVC and decision trees, in addition to time-series baselines, fully-convolutional CNNs, and Vision Transformers. Please refer to the user guidebook for further information.

## Visualizations
We divide our approach to visualizations into two categories, _ad-hoc_ and _post-hoc_. _ad-hoc_ visualizations are produced at runtime from functions in `ml/plot.py` and provide context or results on a single experiment run. Our _post-hoc_ visuals found in `visuals/visuals.py` aggregate results over multiple runs or experiments and produce comparative figures.

If you're looking for a visualization function, it's probably in `ml/plot.py` and is produced at runtime. _post-hoc_ visuals are only necessary when comparing hyperparameter settings on our baseline classification experiment.

## Moving Forward
There are several avenues worth exploring in future studies, as well as several current duties which should be finished prior to the concolusion of the contract. These include, though are not limited to:
+ Refactoring and documenting older code, specifically in `visuals/` and `preprocessing/`. 
+ Improving stability of scripts by addressing active issues.
+ Composing a short (<5 page) document providing instructions and commentary on running ML scripts.
+ Developing a clean format for figures and re-running experiments to obtain them.
+ Implementing a method for batched model loading in experiments.
+ Improving script exporting & logging clarities and directories.
+ Expanding all scripts for more than 4-classes.
+ Investigating angle measurement script and attempting to integrate it with our preprocessing.

Further research can involve:
+ Producing Self-attention Attribution visualizations per [Hao et al](https://arxiv.org/abs/2004.11207)
+ Improving stability of our Transformers by applying Admin initialization scheme per [Liu et al](https://arxiv.org/abs/2004.08249)
