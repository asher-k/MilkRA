import re
import os
import logging

import pandas as pd
import sklearn.preprocessing as sklp
from sklearn.decomposition import PCA


def run_pca(data, labels, seed, num_components=5):
    """
    Performs PCA on the provided data and labels

    :param data: data to transform
    :param labels: associated class labels
    :param seed: emables deterministic results
    :param num_components: Number of principle components to consider; anything 10 > x > 2 performs well
    :return:
    """
    standardizer = sklp.StandardScaler().fit(data)
    dstd = standardizer.transform(data)
    pca = PCA(random_state=seed, n_components=num_components)  # >0.9 @ 2 PCs
    data = pd.DataFrame(pca.fit_transform(dstd, labels))
    logging.info(f"PCA explained variance: {pca.explained_variance_ratio_}")
    return data


def format_name(arg, d=None, ext=None):
    """
    Produces a formatted file name

    :param arg: command-line arguments used to construct file name
    :param d: export directory prepended to name
    :param ext: file extension appended to name
    :return:
    """
    fname = "{save_dir}{name}.{model}.{type}.{norm}.{avg}{only}{ext}"
    fname = fname.format(
        save_dir=d,
        name=arg.name,
        model=arg.model,
        type=arg.type,
        norm=f"norm{arg.normalize}",
        avg="mpmean." if arg.centre_avg else "",
        only=""+str(arg.load_only) if str(arg.load_only) is not None else "",
        ext=ext
    )
    return fname


def load(data_dir, data_type, **kwargs):
    """
    Loads preprocessed samples of droplet sequences. This involves normalization, feature selection, reshaping etc. per
    values of **kwargs

    :return: DataFrame dataset, List labels
    """
    x, y, norm_consts = [], [], []
    classes = os.listdir(data_dir)
    for c in classes:
        class_dir = "{d}/{c}".format(d=data_dir, c=c)
        seqs, index_cols = [], []
        for f in os.listdir(class_dir+"/"):  # individually load each .csv file
            file_dir = "{c}/{f}/".format(c=class_dir, f=f)
            files = os.listdir(file_dir)
            file = list(filter(lambda fl: data_type+".csv" in fl, files))
            file = pd.read_csv(f"{file_dir}{file[0]}")
            index_cols.append(file.columns.get_loc("dl_height_midpoint"))  # track normalization index
            seqs.append(file)

        [norm_consts.append(s.iloc[0, i]) for s, i in zip(seqs, index_cols)]  # update normalization constant
        seqs = [s[_col_order(data_type)] for s in seqs]  # reshape column orders

        if kwargs['only'] is not None:  # row selection; irrelevant rows are discarded
            seqs = [i.iloc[kwargs['only'], :] for i in seqs]
        elif kwargs['at'] is not None:
            seqs = [i.iloc[kwargs['at'], :] for i in seqs]
        elif kwargs['ranges'] is not None:
            ranges = _parse_ranges(kwargs['ranges'])
            seqs = [i.iloc[ranges, :] for i in seqs]
        else:
            seqs = [i.iloc[:900, :] for i in seqs]

        if kwargs['features'] is not None:  # reduce features to indicies
            seqs = [i.iloc[:, kwargs['features']] for i in seqs]

        if kwargs['centre_avg']:  # get mean over centre 3 observations and drop original observations
            to_avg = ["dl_height_midpoint", "2l_to_2r", "1l_to_1r"]
            for i in seqs:
                i["midpoints_mean"] = i[to_avg].mean(axis=1)
            seqs = [i.drop(to_avg, axis=1) for i in seqs]

        seqs = [i.to_numpy().flatten() for i in seqs]  # flatten instances
        x += seqs
        y += [c] * len(seqs)

    x = pd.DataFrame(x)
    x = x.fillna(0)  # 0-imputation

    if kwargs['normalize'] == "max":
        x = x.div(norm_consts, axis=0)
    elif kwargs['normalize'] == "const":
        x = x.div(1000, axis=0)
    return x, y


def _col_order(data_type):
    """
    Returns constants representing column orders defined externally by the data

    :param data_type: type of data being used: either raw or processed
    :return: column orders for the corresponding dataset
    """
    if data_type == "processed":
        return ['edge_4_r_to_edge_4_l', 'edge_3_r_to_edge_3_l', '11l_to_11r', 'edge_2_r_to_edge_2_l',
                'edge_1_r_to_edge_1_l', '10l_to_10r', '9l_to_9r', '8l_to_8r', '7l_to_7r', '6l_to_6r', '5l_to_5r',
                '4l_to_4r', '3l_to_3r', '2l_to_2r', '1l_to_1r', 'dl_height_midpoint']
    return ['edge_4_l', 'edge_3_l', '11l', 'edge_2_l', 'edge_1_l', '10l', '9l', '8l', '7l', '6l', '5l', '4l', '3l',
            '2l', '1l', 'dl_height_midpoint', '1r', '2r', '3r', '4r', '5r', '6r', '7r', '8r', '9r', '10r', 'edge_1_r',
            'edge_2_r', '11r', 'edge_3_r', 'edge_4_r']


def _parse_ranges(ranges, split=":"):
    """
    Parses user-provided row ranges in --load_ranges

    :param ranges: List of string-formatted ranges
    :param split: character to split ranges with; assumed to use default ':'
    :return:
    """
    parsed = set()
    for ur in ranges:
        s = re.split(split, ur)
        assert len(s) == 3
        start, end, step = (int(i) for i in s)
        parsed.update(set(range(start, end, step)))
    return sorted(parsed)
