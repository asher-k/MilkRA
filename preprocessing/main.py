# Main script; always run this in place of preprocess.py & script.py!!!
from argparse import ArgumentParser, BooleanOptionalAction
from crop import crop_all
import preprocess as single
import script as multi
from Image import CONSTS

    
def define_arguments():
    """
    Establishes default model & training/inference parameters.

    :return: default arguments
    """
    a = ArgumentParser()
    a.add_argument('--mode', default='single', type=str, help='Mode of the script; one of \'single\' or \'multi\'')
    a.add_argument('--datapath', default='../data', type=str, help='Path to the data folder')
    a.add_argument('--csv_exprpath', default='../output/csv', type=str, help='Path to export .csv to')
    a.add_argument('--img_exprpath', default='../output/annotations', type=str, help='Path to export .tiff files to')
    a.add_argument('--annotate', default=False, action=BooleanOptionalAction,
                   help='Export annotated images with reflection & height?')

    a.add_argument('--dataset', default=None, type=str, help='Folder to evaluate on (single only)')
    a.add_argument('--crop', default=False, action=BooleanOptionalAction,
                   help='Automatically crop and/or used cropped versions of imgs?')
    a.add_argument('--croppath', default='../cropped', type=str, help='Path to the folder containing cropped images')
    a.add_argument('--height_radius', default='2', type=int, help='Radius when calculating the height of the droplet')
    a.add_argument('--height_method', default='top', type=str, help='Method used to calculate height, top or bottom')

    a.add_argument('--width_only',  default=False, action=BooleanOptionalAction,
                   help='Special script mode; enables dynamic width tracking per timestep and removes height tracking')
    a = a.parse_args()
    return a


# Delegate model to correct mode, using the provided arguments.
if __name__ == '__main__':
    args = define_arguments()
    CONSTS.HEIGHT_RADIUS = args.height_radius

    # First check for crop
    if args.crop:
        crop_all(args.datapath, args.dataset, args.mode == 'single', args.croppath)
        args.datapath = args.croppath  # Update the directory to that of the cropped images

    # Then move onto mode
    if args.mode == 'single':
        single.run(args.datapath, args.dataset, args.csv_exprpath, args.img_exprpath, args.annotate, args.height_method,
                   width_only=args.width_only)
    elif args.mode == 'multi':
        multi.run(args.datapath, args.csv_exprpath, args.img_exprpath, args.annotate, args.height_method)
    else:
        raise RuntimeError('Unknown script mode: %s, mode should either be \'single\' or \'multi\'' % args.mode)
