from argparse import ArgumentParser


def define_arguments():
    """
    Establishes default model & training/inference parameters.

    :return: default arguments
    """
    a = ArgumentParser()
    a.add_argument('--mode', default='logreg', type=str, help='Mode ')
    a = a.parse_args()
    return a


def load():
    """
    Loads preprocessed samples of droplet sequences.

    :return:
    """
    data = []
    return data


# Delegate model to correct mode, using the provided arguments.
if __name__ == '__main__':
    # Load droplet data

    pass
