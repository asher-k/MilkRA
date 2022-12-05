from argparse import ArgumentParser


def define_arguments():
    """
    Establishes default model & training/inference parameters.

    :return: default arguments
    """
    a = ArgumentParser()
    a.add_argument('--mode', default='sc', type=str, help='')
    a = a.parse_args()
    return a


# Delegate model to correct mode, using the provided arguments.
if __name__ == '__main__':
    # TODO
    pass
