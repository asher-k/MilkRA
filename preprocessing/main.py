# Main script; always run this in place of preprocess.py & script.py!!!
from argparse import ArgumentParser
from crop import crop_all
import preprocess as single
import script as multi


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
    a.add_argument('--annotate', default=False, type=bool, help='Export annotated images with reflection & height?')

    a.add_argument('--dataset', default=None, type=str, help='Folder to evaluate on (single only)')
    a.add_argument('--crop', default=False, type=bool, help='Automatically crop each image in the imageset to the '
                                                            'droplet?')
    a = a.parse_args()
    return a


# Delegate model to correct mode, using the provided arguments.
if __name__ == '__main__':
    args = define_arguments()

    # First check for crop
    if args.crop:
        crop_all()
    # Then move onto mode
    if args.mode == 'single':
        single.run(args.datapath, args.dataset, args.csv_exprpath, args.img_exprpath, args.annotate)
    elif args.mode == 'multi':
        multi.run(args.datapath, args.csv_exprpath, args.img_exprpath, args.annotate)
    else:
        raise RuntimeError('Unknown script mode: %s, mode should either be \'single\' or \'multi\'' % args.mode)
