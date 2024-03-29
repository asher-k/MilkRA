import re
import os
import logging
import plots
import torch
import umap
import numpy as np
import pandas as pd
import sklearn.preprocessing as sklp
from sklearn.decomposition import PCA
from torch.utils.data import Dataset


class DropletDataset(Dataset):
    """
    Structured dataset for Torch models. Enables retrieval of droplet labels and droplet time-series data and allows
    for modification through the use of transformations.
    """
    def __init__(self, X, y, v, id, transforms=None):
        self.droplets_frame = X
        self.droplets_label = y
        self.droplets_volumes = v
        self.droplets_id = id
        self.transforms = transforms

    def __len__(self):
        """
        Computes the number of items in the Dataset.

        :return: Int
        """
        return len(self.droplets_frame)

    def __getitem__(self, idx):
        """
        Retrieves and transforms the Dataset item at the provided index.

        :param idx: Int index to retrieve item at
        :return: Tensor item at idx, transformed by all assigned transformations
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.droplets_frame[idx]
        for transform in self.transforms:
            sample = transform(sample)

        label = self.droplets_label[idx]
        label = _label_to_index(label)

        id = self.droplets_id[idx]
        vol = self.droplets_volumes[idx]
        return sample, label, id, vol

    def labels(self):
        """
        Dynamic computation of set and number of distinct labels in the dataset.

        :return: Tupe(List, Int) unique labels, number of labels
        """
        labels = set(self.droplets_label)
        return list(labels), len(labels)


class ToTensor(object):
    """
    Transformation that converts our 3d NumPy arrays into Tensors.
    """
    def __call__(self, sample):
        """
        Converts a NumPy sample to a Tensor.

        :param sample: NumPy sample to convert
        :return: Converted sample
        """
        sample = torch.from_numpy(sample)
        return sample


class FloatTransform(object):
    """
    Transformation that converts all values in a 3d numpy array to float32 to ensure Torch compatability.
    """
    def __call__(self, sample):
        """
        Performs the float conversion.

        :param sample: NumPy sample to convert
        :return: Converted sample
        """
        sample = np.array([np.array([np.array([np.float32(val) for val in r]) for r in c]) for c in sample])
        return sample


class SubdivTransform(object):
    """
    Subdivides the provided image into N windows, discarding any incomplete windows. If a window size is not provided it
    is assumed to be square
    """
    def __init__(self, ss=None, flatten=True):
        self.subdiv_size = ss
        self.flatten = flatten

    def __call__(self, sample, axis=2):
        """
        Performs the image subdivision.

        :param sample: Tensor sample to convert
        :param axis: Axis to perform the subdivision across
        :return: NumPy array of the subdivided image
        """
        if self.subdiv_size is None:
            self.subdiv_size = sample.shape[axis]  # Divides image into equal-sized square subdivisions
        n_subdivs = sample.shape[1] // self.subdiv_size
        sample = np.array([sample[:, self.subdiv_size*n:self.subdiv_size*(n+1), :] for n in range(0, n_subdivs)])
        sample = np.squeeze(sample)  # drop channels
        if self.flatten:  # Flatten C, H, W to C, H*W
            sample = np.reshape(sample, (sample.shape[0], sample.shape[1]*sample.shape[2]))
        return sample


def run_pca(X, y, seed, out_dir, num_components=2, verbose=False, fname=""):
    """
    Performs PCA on the provided data and labels and displays a plot of the transformed points.

    :param X: Droplet data to transform
    :param y: Droplet class labels
    :param seed: For deterministic results
    :param out_dir: Output directory for embedding visualization
    :param num_components: Number of principle components to consider; anything 10 > x > 2 performs well, with 5 being +
    :param verbose: Enables the display of a plot of the embedding (requires n_components = 2)
    :param fname: Substringh appended as a file name

    :return: Transformed X data, auxiliary verbosity output
    """
    standardizer = sklp.StandardScaler().fit(X)
    dstd = standardizer.transform(X)
    pca = PCA(random_state=seed, n_components=num_components)  # >0.9 @ 2 PCs
    X = pd.DataFrame(pca.fit_transform(dstd, y))
    if verbose:
        logging.info(f"PCA explained variance: {pca.explained_variance_ratio_}")
        plots.plot_embedding_visualization(X, y, out_dir, method=f"PCA_{fname}")
    return X, pca.explained_variance_ratio_


def run_umap(X, y, seed, out_dir, num_components=2, verbose=False):
    """
    Performs UMAP dimensionality reduction and displays a plot of the transformed points.

    :param X: Droplet data to transform
    :param y: Droplet class labels
    :param seed: For deterministic results
    :param out_dir: Output directory for embedding visualization
    :param num_components: Number of principle components to consider; anything 10 > x > 2 performs well, with 5 being +
    :param verbose: Enables the display of a plot of the embedding (requires n_components = 2)
    :return: Transformed X data
    """
    umapper = umap.UMAP(n_components=num_components, random_state=seed)
    X_mapped = umapper.fit_transform(X)
    if verbose:
        plots.plot_embedding_visualization(X_mapped, y, out_dir, method="UMAP")
    return X_mapped


def format_name(arg, d=None, ext=None):
    """
    Produces a formatted file name by prepending experiment parameters taken from command-line arguments and the export
    directory while appending the file extension.

    :param arg: Command-line arguments used to construct file name
    :param d: Export directory prepended to name
    :param ext: File extension appended to name
    :return: The final file name
    """
    fname = "{save_dir}{name}.{model}.{type}.{norm}.{avg}{ext}"
    fname = fname.format(
        save_dir=d,
        name=arg.name,
        model=arg.model,
        type=arg.type,
        norm=f"norm{arg.normalize}",
        avg="mpmean." if arg.centre_avg else "",
        ext=ext
    )
    return fname


def load(data_dir, data_type, **kwargs):
    """
    Loads preprocessed samples of droplet sequences from the provided directory and data type. Normalization, feature
    selection, reshaping etc. are enabled through kwargs.

    :param data_dir: Directory to read droplet samples from
    :param data_type: String either "processed" or "raw"

    :return: DataFrame dataset, List labels
    """
    x, y, norm_consts = [], [], []
    classes = sorted(os.listdir(data_dir))
    volume = classes.pop()
    for c in classes:
        class_dir = "{d}/{c}".format(d=data_dir, c=c)
        seqs, index_cols = [], []
        files = sorted(os.listdir(class_dir+"/"))
        for f in files:  # load each sample's .csv file sequentially
            file_dir = "{c}/{f}/".format(c=class_dir, f=f)
            files = os.listdir(file_dir)
            file = list(filter(lambda fl: data_type+".csv" in fl, files))
            file = pd.read_csv(f"{file_dir}{file[0]}")
            index_cols.append(file.columns.get_loc("dl_height_midpoint"))  # track normalization index
            seqs.append(file)

        [norm_consts.append(s.iloc[0, i]) for s, i in zip(seqs, index_cols)]  # update normalization constant
        seqs = [s[_col_order(data_type)] for s in seqs]  # reshape column orders according to dataset used

        if kwargs['at'] is not None:  # row selection; irrelevant rows are discarded
            seqs = [i.iloc[kwargs['at'], :] for i in seqs]
        elif kwargs['ranges'] is not None:
            ranges = _parse_ranges(kwargs['ranges'])
            seqs = [i.iloc[ranges, :] for i in seqs]
        elif kwargs['timesteps'] is not None:
            Ts = np.load(kwargs['timesteps'])
            Ts = np.where(Ts == 1)[0]
            seqs = [i.iloc[Ts, :] for i in seqs]
        else:
            seqs = [i.iloc[:900, :] for i in seqs]

        if kwargs['features'] is not None:  # reduce features to selected indicies
            seqs = [i.iloc[:, kwargs['features']] for i in seqs]
        if kwargs['centre_avg']:  # get mean over centre 3 observations and drop original observations
            to_avg = ["dl_height_midpoint", "2l_to_2r", "1l_to_1r"]
            for i in seqs:
                i["midpoints_mean"] = i[to_avg].mean(axis=1)
            seqs = [i.drop(to_avg, axis=1) for i in seqs]

        if not kwargs['ts']:
            seqs = [i.to_numpy().flatten() for i in seqs]  # flatten instances in the non time-series case
        x += seqs
        y += [c] * len(seqs)

    # Different formatting for time-series & non-ts data
    if kwargs['ts']:
        x = [pd.DataFrame(i) for i in x]
        for e, i in enumerate(x):
            i = i.fillna(0)  # 0-imputation
            if kwargs['normalize']:
                x[e] = i.div(norm_consts[e], axis=0)
        return np.array([i.transpose().to_numpy() for i in x]), np.array(y)
    else:
        x = pd.DataFrame(x)
        x = x.fillna(0)  # 0-imputation
        if kwargs['normalize'] == "max":
            x = x.div(norm_consts, axis=0)
        elif kwargs['normalize'] == "const":
            x = x.div(1000, axis=0)
        return x, y


def load_volumes(path):
    """
    Loads the droplet volumes from a .csv file.

    :param path: Path to the csv file
    :return: Numpy array composed of the index and volume of a droplet
    """
    return pd.read_csv(path).to_numpy()


def _col_order(data_type):
    """
    Returns constants representing column orders defined externally by the data.

    :param data_type: Type of data being used: either raw or processed.
    :return: List column orders for the corresponding dataset
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
    Parses user-provided timestep ranges in the command-line argument --load_ranges.

    :param ranges: List of string-formatted ranges
    :param split: Character to split ranges with; assumed to use default ':'
    :return: List of numeric ranges sorted
    """
    parsed = set()
    for ur in ranges:
        s = re.split(split, ur)
        assert len(s) == 3
        start, end, step = (int(i) for i in s)
        parsed.update(set(range(start, end, step)))
    return sorted(parsed)


def _label_to_index(label):
    """
    Converts string label of a sample to a numeric representation.

    :param label: String label of the data; assumed to be unaltered from the four classes provided in the data directory
    :return: Int representation of the label
    """
    map_dict = {"DBM 1000mA Repeats": 0, "GTM 1000mA Repeats": 1, "LBM 1000mA Repeats": 2, "LBP+ 1000mA Repeats": 3}
    return map_dict[label]
