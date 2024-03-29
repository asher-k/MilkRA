import os
import torch
import logging
import pandas as pd
import numpy as np
import numpy.random as nprand
import pyswarms as ps
import nn as nets
import transformer as trans
import plots as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch import nn
from tqdm import tqdm
from models import Baselines, Clustering, TSBaselines
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.model_selection import train_test_split
from data import format_name, _col_order, run_pca, run_umap, DropletDataset, ToTensor, FloatTransform, SubdivTransform, load_volumes
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.feature_selection import f_classif
from warnings import simplefilter


def classify_baselines(args, X, y, out_dir):
    """
    Classification experiment on baseline non-time series models

    :param args: Command-line ArgParser
    :param X: Droplet data
    :param y: Droplet classes
    :param out_dir: Subdirectory to export logs & generate files to
    """
    baselines = Baselines()

    for model in baselines.m[args.model]:
        for row in X.index:  # inits samplewise misclassification counts
            baselines.preddict[row] = (0., 0)

        # can infer model name from baseline classifier keys
        if args.model == "all":
            model_name = list(baselines.m.keys())[baselines.m[args.model].index(model)]
        else:
            model_name = list(baselines.m.keys())[list(baselines.m.values()).index([model])]

        X_selected = X.copy()
        if args.features_selection == "top":  # since Percentile is deterministic no need to run @ each seed
            selector = SelectPercentile(score_func=mutual_info_classif, percentile=20)
            selector.fit(X_selected, y)
            X_selected = pd.DataFrame(selector.transform(X_selected))

        # obtain model results over n seeds
        r, cm, p, dt_top = [], [], [], []  # results, confusion matricies, feature importance, decision tree splits
        for state in nprand.randint(0, 999999999, size=args.num_states):
            nprand.seed(state)
            state_data = X_selected.copy()
            if args.features_selection == "pca":  # PCA can be deterministic under randomizer solver; may occur in BD
                state_data, _ = run_pca(state_data, y, state, out_dir=f"{out_dir}figs/", verbose=True)
            elif args.features_selection == "umap":
                state_data = run_umap(state_data, y, state, out_dir=f"{out_dir}figs/", verbose=True)

            train_d, test_d, train_l, test_l = train_test_split(state_data, y, test_size=0.3, stratify=y, random_state=state)
            baselines.data(train_d, train_l, test_d, test_l)
            result, c, _imp, _split = model(random_state=state,
                                            verbose=False,  # args.verbose
                                            importance=args.importance,
                                            feature_names=_col_order(args.type),
                                            only_acc=args.only_acc,
                                            out_dir=f"{out_dir}figs/")
            for tr, re in zip([r, cm, p, dt_top], [result, c, _imp, _split]):
                tr.append(re)

        # aggregate results over all seeds & log it!
        results = pd.DataFrame(r)
        stddevs = results.std()  # light post-run averaging
        results = results.mean(axis=0).round(decimals=3)
        for col, val in stddevs.items():  # reformatting stddev column names
            results[col+"+-"] = val
        results = results.to_frame().transpose()
        results = results.reindex(sorted(results.columns), axis=1)
        logging.info(msg="\nPerformance Statistics: {mod}\n{res}\n".format(mod=model_name, res=str(results)))

        # save performance results to csv file(s) with any produced plots
        if args.save:
            # Export CSVs to sub-directory
            results_dir = f"{out_dir}results/"
            results.to_csv(f"{results_dir}{model.__name__}.csv")

        if args.verbose:  # aggregate confusion matrix & misclassification rate figures & save to output directory
            plt.plot_samplewise_misclassification_rates(baselines, 22, y, args, out_dir, model_name)
            plt.plot_confusion_matrix(cm, y, out_dir, model_name)

            if args.importance and not args.only_acc and model_name in ["logreg", "dt"]:  # format feature importances
                importance_dir = "{sd}importance/".format(sd=out_dir)
                if not os.path.exists(importance_dir):
                    os.mkdir(importance_dir)
                priority = pd.DataFrame(p, columns=_col_order(args.type))
                priority = priority.abs()
                cvs = np.cov(priority, rowvar=False)
                imp_name = format_name(args, importance_dir, ".npy")
                np.save(imp_name, cvs)
                priority = priority.mean(axis=0).round(decimals=3).to_frame().transpose()
                priority = priority.to_frame().transpose()
                imp_name = format_name(args, importance_dir, ".csv")
                priority.to_csv(imp_name)

                if model_name == "dt":  # Also save decision tree splits for analysis
                    importance_splits_dir = "{id}splits/".format(id=importance_dir)
                    if not os.path.exists(importance_splits_dir):
                        os.mkdir(importance_splits_dir)
                    dt_splits = pd.DataFrame(dt_top, columns=["top_split"])
                    imp_name = format_name(args, importance_splits_dir, ".csv")
                    dt_splits.to_csv(imp_name)  # need to update save dir...


def classify_ts(args, X, y, out_dir):
    """
    Classification experiment with Time-series models.

    :param args: Command-line ArgParser
    :param X: Droplet data
    :param y: Droplet classes
    :param out_dir: Sub-directory to save any produced files in
    """
    models = TSBaselines()
    nprand.seed(args.seed)
    for model in models.m[args.model]:
        r, cm = [], []  # results, confusion matricies
        for i, s in enumerate(nprand.randint(0, 999999999, size=args.num_states)):
            model_name = f"{i}_{s}"
            nprand.seed(s)
            logging.info(f"Beginning {args.model} model {model_name}")

            X_copy = X.copy()
            train_d, test_d, train_l, test_l = train_test_split(X_copy, y, test_size=args.pyt_data_split[1], stratify=y)
            models.data(train_d, train_l, test_d, test_l)
            result, c = model(random_state=s, verbose=True)
            r.append(result)
            cm.append(c)

        # aggregate seed results & log
        results = pd.DataFrame(r)
        stddevs = results.std()  # light post-run averaging
        results = results.mean(axis=0).round(decimals=3)
        for col, val in stddevs.items():  # reformatting stddev column names
            results[col + "+-"] = val
        results = results.to_frame().transpose()
        results = results.reindex(sorted(results.columns), axis=1)
        logging.info(results)

        # Export results to sub-directory
        results_dir = f"{out_dir}results/"
        results.to_csv(f"{results_dir}{model.__name__}.csv")

        # display & save confusion matrix
        if args.verbose:
            plt.plot_confusion_matrix(cm, y, out_dir, mname=args.name)


def clustering(args, X, y, out_dir):
    """
    Clustering experiment on our PCA'd data.

    :param args: Command-line ArgParser
    :param X: Droplet data
    :param y: Droplet classes
    :param out_dir: Sub-directory to save any produced files in
    """
    clusters = Clustering()
    if args.features_selection == "pca":  # PCA can be deterministic under randomizer solver; may occur in BD
        X, _ = run_pca(X, y, args.seed)

    clusters.data(X, y)
    model = clusters.m[args.model][0]
    model = model()
    clusters.plot_dendrogram(model, out_dir)


def pso(args, X, y, out_dir):
    """
    PSO-driven timestep selection tuning via PCA.

    :param args: Command-line ArgParser
    :param X: Droplet data
    :param y: Droplet classes
    :param out_dir: Sub-directory to save any produced files in
    """
    def prop_check(props, ind, seg_size):
        # checks that we can still add it to the segment without going over the selected proportion of features
        segment = ind // seg_size
        prop_change = 1 / seg_size
        props = np.array([p if i != segment else p-prop_change for i, p in enumerate(props)])
        return props[segment] > 0, props

    def prop_to_bin(x, appr=args.pso_proportional_appr):
        segment_size = X.shape[2] // len(x)
        bin_x = []
        if appr == 'None':  # Use denominator step-based approach
            for prop in x:
                segment = list(np.zeros(segment_size, dtype=int))
                if prop != 0.0:
                    interval = int(segment_size / max(int(segment_size*prop), 1))
                    n_selected = len(segment[0::interval])
                    segment[0::interval] = list(np.ones(n_selected, dtype=int))
                bin_x += segment  # add the segment to the full binary feature array
        elif appr == 'MI':   # MI-based approach
            for prop, segment in zip(x, [ami[n*segment_size:(n+1)*segment_size] for n in range(0, len(ami))]):
                min_mi = list(reversed(sorted(segment)))[int(prop*segment_size)]
                segment = [1 if mi > min_mi else 0 for mi in segment]
                bin_x += segment
        else:       # MRMR-based approach
            fpr = X.shape[1]  # features per row (needed for correct indexing of correlations)
            ids = []          #  indices of selected timesteps
            mrmrs = np.nan_to_num(f_score.reshape((900 * 16)) * corr.to_numpy())  # Computed MRMR scores for each feature/feature pair

            # Add the first feature by recording its index in the sequence and removing it
            next_ind = np.argmax(np.max(f_score, axis=1))
            ids.append(next_ind)
            x = prop_check(x, next_ind, segment_size)[1]  # Reduce the needed proportions
            current_mrmrs = mrmrs[:][ids[-1]*fpr:(ids[-1]+1)*fpr]  # update the subsection we evaluate from
            finished_segments = x < (1 / segment_size)  # sections which have reached 0% more proportion needed

            # Loop; while there are still segments that require a value to add to
            while any([v > 1/segment_size for v in x]):
                summd_mrmrs = np.sum(current_mrmrs, axis=0)  # first sum over all features for total relevancy
                summd_mrmrs = np.sum(np.reshape(summd_mrmrs, (900, 16)), axis=1)  # Then sum over again for timestep relevancy
                np.put(summd_mrmrs, ids, 0)  # remove any timesteps we've already selected
                summd_mrmrs = np.array([0 if c else a for a, c in zip(summd_mrmrs, np.repeat(finished_segments, segment_size))])
                sortd_mrmrs = list(np.argsort(summd_mrmrs))

                found = False
                while not found:  # Loop; until we find a value in a valid segment
                    next_ind = sortd_mrmrs.pop()  # Get index of the next best timestep
                    found, props = prop_check(x, next_ind, segment_size)

                ids.append(next_ind)
                x = props  # Reduce the needed proportions
                current_mrmrs = np.append(current_mrmrs, mrmrs[:][ids[-1] * fpr:(ids[-1] + 1) * fpr], axis=0)  # update the subsection we evaluate from
                finished_segments = x < (1 / segment_size)
            bin_x = np.zeros(900)
            np.put(bin_x, ids, 1)
            assert np.count_nonzero(bin_x) == len(ids)
        return np.array(bin_x)

    def pca_reshape(x, f, flatten=True):
        if len(f) != X.shape[2]:  # Accounts for proportion-based selection in place of binary
            f = prop_to_bin(f)
        x_selected = np.array([a[:, f == 1] for a in x])
        shape = x_selected.shape
        if flatten:
            x_selected = x_selected.reshape((shape[0], shape[1] * shape[2]))
        return x_selected

    simplefilter(action='ignore', category=FutureWarning)
    def pcac_loss(f, use_ensemble=True):
        """
        PCA-Clustering loss function.

        :param f: Iterable of binary features
        :return: Loss term
        """
        x_selected = pca_reshape(X, f)
        if use_ensemble:  # Classifier-based loss
            m = KNeighborsClassifier()
            m.fit(x_selected, y)
            preds = m.predict(x_selected)
            return (1-accuracy_score(preds, y)) + (0.00001 * (x_selected.shape[1] / X.shape[1]))
        else:  # PCA-Clustering loss
            x_selected, score = run_pca(x_selected, y, args.seed, out_dir=out_dir)
            unaccounted_var = 1 - sum(score)  # PCA error

            inverse_silhouette = silhouette_score(x_selected, y)
            inverse_silhouette = 1 / inverse_silhouette  # == 1 means defined clusers, > 1 means silhouette -> 0 & less def.
            if args.verbose:
                logging.info(f"PCA loss: {unaccounted_var}, Silhouette: {inverse_silhouette}")
            return unaccounted_var + inverse_silhouette

    def batch_loss(fs):
        losses = [pcac_loss(f) for f in fs]
        return np.array(losses)

    aggr_proportions, aggr_subsets = [], []
    for seed in nprand.randint(0, 10000000, size=args.num_states):
        nprand.seed(seed)
        if not args.load:  # only implemented direct experiments for now, no loading/re-testing
            if args.pso_type == 'binary':
                n_particles, n_dims = X.shape[2]**2//args.pso_prop, X.shape[2]
                assert args.pso_initsize < n_dims
                init_ps = None
                if args.pso_initscheme == 'deterministic':
                    init_ps = np.zeros(shape=(n_particles, n_dims))
                    for particle in init_ps:
                        inds = nprand.choice(list(range(0, n_dims-1)), size=args.pso_initsize)
                        particle[inds] = 1
                else:
                    init_ps = nprand.choice([0,1], size=(n_particles, n_dims),
                                             p=[1.-(args.pso_initsize/n_dims), args.pso_initsize/n_dims])
                    for particle in init_ps:  # verify we never have an "empty" particle
                        if not any(particle):
                            particle[nprand.choice(list(range(0, n_dims)))] = 1

                settings = {'c1': 2.5, 'c2': 0.5, 'w': 0.2, 'k': 20, 'p': 2}  # cognitive, social, inertia, n neighbours, distance metric
                optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=n_dims, options=settings, init_pos=init_ps)
                cost, pos = optimizer.optimize(batch_loss, iters=args.pso_iters)

            elif args.pso_type == 'proportional':
                if args.pso_proportional_appr == 'MI':
                    X_flat = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
                    ami = mutual_info_classif(X_flat, y)
                    ami = np.mean(ami.reshape((-1, X.shape[2])), axis=0)
                    plt.plot_pso_scores_at_timesteps(ami, out_dir, "MI Score", "figs/MI_timesteps")

                    n_particles, n_segments = 150, 10
                    settings = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
                    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=n_segments,
                                                        options=settings,
                                                        bounds=(np.zeros(n_segments), np.ones(n_segments) * 0.1))
                    cost, pos = optimizer.optimize(batch_loss, iters=args.pso_iters)
                    aggr_proportions.append(pos)
                    pos = prop_to_bin(pos)  # convert from proportions to binary
                    aggr_subsets.append(pos)
                elif args.pso_proportional_appr == 'MRMR':
                    # First compute f-scores of each timestep variable
                    logging.info("Computing F-scores and feature correlations. This may take a sec...")
                    X_flat = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
                    f_score = np.nan_to_num(f_classif(X_flat, y)[0].reshape(X.shape[2], X.shape[1]))
                    var = np.nan_to_num(np.var(X_flat, axis=0)).reshape(X.shape[2], X.shape[1])
                    plt.plot_pso_scores_at_timesteps(np.average(f_score, axis=1), out_dir, "F Score", "figs/F_timesteps")
                    plt.plot_pso_scores_at_timesteps(np.average(var, axis=1), out_dir, "Mean Var", "figs/var_timesteps")
                    corr = 1-pd.DataFrame(X_flat).corr().abs().clip(1e-5)

                    n_particles, n_segments = 50, 10
                    settings = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
                    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=n_segments,
                                                        options=settings,
                                                        bounds=(np.zeros(n_segments), np.ones(n_segments) * 0.1))
                    cost, pos = optimizer.optimize(batch_loss, iters=args.pso_iters)
                    logging.info(pos)
                    aggr_proportions.append(pos)
                    pos = prop_to_bin(pos)  # convert from proportions to binary
                    aggr_subsets.append(pos)
                else:
                    raise ValueError(f"Unrecognized feature selection method for proportional pso: {args.pso_proportional_appr}.")

                # Display final plots & PCA
                x_selected = pca_reshape(X, pos, flatten=False)
                plt.plot_sample_vs_mean(x_selected, y, [0, 15, 40, 55], out_dir)  # show example images
                x_selected = pca_reshape(X, pos, flatten=True)
                logging.info(
                    f"# features: {(x_selected.shape[1] / X.shape[1])} MisCl:{cost - (0.00001 * (x_selected.shape[1] / X.shape[1]))}")
                x_selected, _ = run_pca(x_selected, y, args.seed, out_dir=out_dir, verbose=True, fname=seed)  # show PCA
                logging.info(f"Final Inverse Silhouette: {1 / silhouette_score(x_selected, y)}")

            elif args.pso_type =='greedy':
                aggr_size = []
                aggr_obj = []
                # not actually PSO, just a greedy search using MRMR selection
                # NOTE: for this you need to run --num_states 1 since we already enumerate each possible starting step.
                fpr = X.shape[1]  # features per row
                X_flat = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
                f_score = f_classif(X_flat, y)[0].reshape(X.shape[2], X.shape[1])
                corr = 1 - pd.DataFrame(X_flat).corr().abs().clip(1e-5)
                mrmrs = np.nan_to_num(f_score.reshape((900 * 16)) * corr.to_numpy())

                end_criteria = [48, 0.05]  # End criteria for a single subset
                for starting_step in list(range(0, 900)):  # enumerate all possible starting timesteps
                    subset = [starting_step]  # start the current subset
                    bin_subset = np.zeros(900)  # binary representation
                    np.put(bin_subset, subset, 1)
                    current_mrmrs = mrmrs[:][subset[-1] * fpr:(subset[-1] + 1) * fpr] # update the subsection we evaluate from

                    while len(subset) < end_criteria[0] and pcac_loss(bin_subset) > end_criteria[1]:
                        summd_mrmrs = np.sum(current_mrmrs, axis=0)
                        summd_mrmrs = np.sum(np.reshape(summd_mrmrs, (900, 16)), axis=1)
                        np.put(summd_mrmrs, subset, 0)  # remove any timesteps we've already selected
                        sortd_mrmrs = list(np.argsort(summd_mrmrs))  # sort features by relevance

                        subset.append(sortd_mrmrs.pop())  # update the subset
                        current_mrmrs = np.append(current_mrmrs, mrmrs[:][subset[-1] * fpr:(subset[-1] + 1) * fpr], axis=0)
                        bin_subset = np.zeros(900)  # update binary representation
                        np.put(bin_subset, subset, 1)
                    # Update tracking
                    aggr_subsets.append(bin_subset)
                    aggr_size.append(len(subset))
                    aggr_obj.append(pcac_loss(bin_subset))
                    # Log & export the subset
                    logging.info(f"Objective on timestep {starting_step}: {aggr_obj[-1]}, {aggr_size[-1]}")
                    out_name = f"{seed}_{starting_step}_features.npy"
                    np.save(f"{out_dir}results/{out_name}", bin_subset)
            else:
                raise ValueError(f"Unrecognized PSO type: {args.pso_type}.")
            if args.save:
                out_name = f"{args.name}_{seed}_features.npy"
                np.save(f"{out_dir}results/{out_name}", pos)
                logging.info(f"Exported feature array to {out_name}")
    if len(aggr_proportions) > 0:
        logging.info(f"Aggregated Proportions: {np.mean(aggr_proportions, axis=0)}")  # display aggregated proportions
    if args.verbose:  # export features chosen to a figure for overview
        plt.plot_pso_subset_counts(np.sum(aggr_subsets, 0), 24, f"{out_dir}figs/", f"{args.name}_top_feature_counts")
        if args.pso_type == 'greedy':
            plt.plot_greedy_search_3d(list(range(0, 900)), aggr_obj, aggr_size, f"{out_dir}figs/", "greedy_performance")


def classify_dl(args, X, y, out_dir):
    """
    Classification experiment with DL models.

    :param args: Command-line ArgParser
    :param X: Droplet data
    :param y: Droplet classes
    :param out_dir: Sub-directory to save any produced files in
    """
    lr, bs, E, spl = args.pyt_lr, args.pyt_bs, args.pyt_epochs, args.pyt_data_split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    performances = {"train_acc": [], "val_acc": []}  # Track performance across multiple seeds
    models = []
    cams = {}  # Track aggregated CAMs across multiple seeds

    # Reformat data & define DataSet
    X_data = np.array([np.array([np.rot90(x, k=3)]) for x in X])  # rotate so we have (time, pos) for (H, W)
    y_data = np.array(y)
    v_data = load_volumes(f"{args.dir}/Volume.csv").swapaxes(0, 1)  # load volume data from separate csv
    v_data = v_data[:][1]  # trim the ids
    if args.pyt_use_volumes:
        logging.info("Using volumes as additional input parameter for final dense layer.")

    data = DropletDataset(X_data, y_data, v_data, list(range(0, len(y_data))), transforms=[
        ToTensor(), FloatTransform(),
    ])
    misc_rates = {i: [] for _x, _y, i, _v in data}  # Misclassification rates of test samples

    tr_size, val_size = int(data.__len__() * spl[0]) + 1, int(data.__len__() * spl[1])

    for index, seed in enumerate(nprand.randint(0, 999999999, size=args.num_states)):
        # Sub-seed setup
        seed = int(seed)
        nprand.seed(seed)
        torch.manual_seed(seed)
        logging.critical(f"Set Seed {index+1}, {seed}")
        model_name = f"{index}_{seed}"

        # Initialize training & test DataLoaders
        (trainData, testData) = random_split(data, [tr_size, val_size], generator=torch.Generator().manual_seed(seed))
        trainLoader = DataLoader(trainData, batch_size=bs, shuffle=True)
        testLoader = DataLoader(testData, batch_size=bs, shuffle=True)
        trainSteps, valSteps = len(trainLoader.dataset) // bs, len(testLoader.dataset) // bs

        # Initialize model, optimizer and logs
        ks = 3 if args.type == "processed" else 5  # adaptive kernel size
        model = nets.CMapNN(num_classes=data.labels()[1], kernel_size=ks, use_volume=args.pyt_use_volumes).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        loss_fn = nn.NLLLoss()
        performance_log = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        conevolution = {}  # convolution + evolution lol
        logging.info(f"Model Size: {nets.count_params(model)}")

        if not args.load:  # Training & validation
            logging.critical(f"Beginning training on model {model_name}")
            for e in (bar := tqdm(range(0, E))):
                performance_log = _dl_train_epoch(model, trainLoader, device, loss_fn, optimizer, performance_log)
                performance_log = _dl_validate(model, testLoader, device, loss_fn, performance_log)
                t, v = performance_log["train_acc"][e], performance_log["val_acc"][e]
                bar.set_description(f"t{t} v{v}")  # Update progress bar
                if args.verbose:  # track convolutions for animation
                    conevolution[e] = [np.copy(model.conv1.weight.cpu().detach().numpy())]  # First conv.; requires input_d 1
        else:  # Load from checkpoint & obtain final performances
            logging.critical(f"Loading model {model_name}...")
            try:
                model.load_state_dict(torch.load(f"{out_dir}models/model{model_name}.pt"))
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not load model {out_dir}models/model{model_name}.pt. "
                                        f"Are you sure the model was saved?")
            for log_type in ["train", "val"]:  # Update training & validation logs
                performance_log = _dl_validate(model,
                                               trainLoader if log_type == "train" else testLoader,
                                               device, loss_fn, performance_log, log_type=log_type)
        performance_log = _normalize_performance_logs(performance_log, trainSteps, tr_size, valSteps, val_size)
        performances["train_acc"].append(round(performance_log["train_acc"][-1], 3))  # Save final performances of seed
        performances["val_acc"].append(round(performance_log["val_acc"][-1], 3))
        models.append(model)

        # Update Misclassification Rates
        with torch.no_grad():
            model.eval()
            for sample in testLoader:
                x, y, i, _v = sample
                x, y = x.to(device), y.to(device)
                pred, _d = model(x, _v)
                pred = pred.argmax(1).cpu().detach().numpy()
                for pc, ac, n in zip(pred, y, i):
                    misc_rates[n.item()].append(0 if pc == ac else 1)

        # Verbosity-enabled plotting
        if args.verbose:
            # Initialize export sub-directories
            cam_export_dir, ep_export_dir, conv_export_dir = f"{out_dir}figs/CAMs/{model_name}/", f"{out_dir}figs/Epochs/", f"{out_dir}figs/Convs/"
            for d in [cam_export_dir, ep_export_dir, conv_export_dir]:
                if not os.path.exists(d):
                    os.makedirs(d)

            if not args.load:  # Epoch Performances
                plt.plot_epoch_performance(E, performance_log.keys(), ep_export_dir, f"{model_name}",
                                           *[i[1] for i in performance_log.items()])
                # for i, d in enumerate(data):  # Sample-wise CAMs
                #     plt.compute_class_activation_maps(model, d[0], d[1], cam_export_dir, str(i), device, display=True)
                # for cam_metric in [np.mean, np.median, np.var]:  # Aggregated CAMs
                #     if cam_metric not in cams:
                #         cams[cam_metric] = []
                #     cams[cam_metric].append(plt.compute_aggregated_cams(model, data, cam_export_dir, cam_metric, str(seed),
                #                                                         device, display=True))
            # conv_trend = [v[0] for k, v in conevolution.items()]  # Evolution of convolutional filters
            # plt.animate_convolution_by_epoch(conv_trend, f=5, t=E, out_dir=conv_export_dir, fname=f"_{model_name}",
            #                                  title=f"seed:{seed}")
        if args.save:  # save model (can be expensive!)
            torch.save(model.state_dict(), f"{out_dir}models/model{index}_{seed}.pt")

    # Trans-seed plotting
    misc_rates = {str(k): sum(v)/len(v) if len(v) > 0 else 0. for k, v in misc_rates.items()}  # Mean misclassification rates
    plt.plot_samplewise_misclassification_rates(misc_rates, 20, y_data, arguments=None, out_dir=f"{out_dir}", mname="full_conv")

    logging.info(f"Final results on {args.num_states} seeds: {performances}")
    if args.verbose:
        plt.plot_training_validation_performance(performances["train_acc"], performances["val_acc"], out_dir)
        plt.plot_training_validation_heatmap(performances["train_acc"], performances["val_acc"], out_dir, tr_size, val_size)


def classify_vit(args, X, y, out_dir):
    """
    Vision Transformer classification experiment.

    :param args: Command-line ArgParser
    :param X: Droplet data
    :param y: Droplet classes
    :param out_dir: Sub-directory to save any produced files in
    """
    lr, bs, E, spl, subdiv_size = args.pyt_lr, args.pyt_bs, args.pyt_epochs, args.pyt_data_split, args.vit_subdiv_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    performances = {"train_acc": [], "val_acc": []}  # Track performance across multiple seeds
    models = []

    # Reformat data
    X_data = np.array([np.array([np.rot90(x, k=3)]) for x in X])  # rotate so we have (time, pos) for (H, W)
    y_data = np.array(y)
    data = DropletDataset(X_data, y_data, transforms=[
        FloatTransform(), SubdivTransform(subdiv_size, flatten=subdiv_size not in [1, ]), ToTensor(),
    ])
    tr_size, val_size = int(data.__len__() * spl[0]) + 1, int(data.__len__() * spl[1])

    for index, seed in enumerate(nprand.randint(0, 999999999, size=args.num_states)):
        # Sub-seed setup
        seed = int(seed)
        nprand.seed(seed)
        torch.manual_seed(seed)
        logging.critical(f"Set Seed {index+1}, {seed}")
        model_name = f"{index}_{seed}"

        # Initialize training & test DataLoaders
        (trainData, testData) = random_split(data, [tr_size, val_size], generator=torch.Generator().manual_seed(seed))
        trainLoader = DataLoader(trainData, batch_size=bs, shuffle=True)
        testLoader = DataLoader(testData, batch_size=bs, shuffle=True)
        trainSteps, valSteps = len(trainLoader.dataset) // bs, len(testLoader.dataset) // bs

        # Configure model, optimizer, loss and logs
        n_subdivs, subdiv_dims = len(data[0][0]), len(data[0][0][0])
        sd, n_dims, n_heads, n_blocks, n_classes = (n_subdivs, subdiv_dims), args.vit_dims, args.vit_heads, \
                                                   args.vit_blocks, data.labels()[1]
        model = trans.ViT(sd=sd, n_dims=n_dims, n_heads=n_heads, n_blocks=n_blocks, n_classes=n_classes).to(device)
        logging.critical(f"Model Size: {nets.count_params(model)}")

        optimizer = Adam(model.parameters(), lr=lr)
        sr = lr_scheduler.StepLR(optimizer, step_size=E//10, gamma=0.95)
        loss_fn = nn.NLLLoss()
        performance_log = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        # Training & validation
        if not args.load:
            for e in (bar := tqdm(range(0, E))):
                performance_log = _dl_train_epoch(model, trainLoader, device, loss_fn, optimizer, performance_log, lrs=sr)
                performance_log = _dl_validate(model, testLoader, device, loss_fn, performance_log)
                t, v = performance_log["train_acc"][e], performance_log["val_acc"][e]
                bar.set_description(f"t{t} v{v}")  # Update progress bar
            if performance_log["train_acc"][-1] < len(trainData)/2 and performance_log["val_acc"][-1] < len(testData)/2:
                logging.critical(f"{model_name} failed. Discarding results.")
                continue
        else:
            logging.critical(f"Loading model {model_name}...")
            try:
                model.load_state_dict(torch.load(f"{out_dir}models/model{model_name}.pt"))
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not load model {out_dir}models/model{model_name}.pt. "
                                        f"Are you sure the model was saved?")
            for log_type in ["train", "val"]:  # Update training & validation logs
                performance_log = _dl_validate(model,
                                               trainLoader if log_type == "train" else testLoader,
                                               device, loss_fn, performance_log, log_type=log_type)
        performance_log = _normalize_performance_logs(performance_log, trainSteps, tr_size, valSteps, val_size)
        performances["train_acc"].append(round(performance_log["train_acc"][-1], 3))  # Save final performances of seed
        performances["val_acc"].append(round(performance_log["val_acc"][-1], 3))
        models.append(model)

        # Verbosity-enabled plotting
        if args.verbose:
            # Initialize export sub-directories
            ep_export_dir, attn_export_dir = f"{out_dir}figs/Epochs/", f"{out_dir}figs/Attns/"
            for d in [ep_export_dir, attn_export_dir]:
                if not os.path.exists(d):
                    os.makedirs(d)

            if not args.load:  # Epoch Performances
                plt.plot_epoch_performance(E, performance_log.keys(), ep_export_dir, f"{model_name}",
                                           *[i[1] for i in performance_log.items()])
            plt.plot_attention_by_class(model, data, n_blocks, attn_export_dir, fname=model_name)  # attention visuals
        if args.save:  # Save model (expensive!)
            torch.save(model.state_dict(), f"{out_dir}models/model{model_name}.pt")

    successes = len(models)
    logging.info(f"Final results on {successes} successful seeds: {performances}")
    if args.verbose:
        plt.plot_training_validation_heatmap(performances["train_acc"], performances["val_acc"], out_dir, tr_size, val_size)


def _dl_train_epoch(model, loader, device, loss_fn, opt, log, verbose=False, lrs=None):
    """
    Performs one training epoch for the provided NN model.

    :param model: CNN or other PyTorch NN Module to train
    :param loader: DataLoader containing training data
    :param device: Current device to run on
    :param loss_fn: Callable Loss Function; requires inputs in the form of (pred_label, act_label)
    :param opt: Model Optimizer
    :param log: Dict containing training/validation keys for tracking
    :param verbose: Enables logging/plotting of the model
    :param lrs: Learning rate scheduler (if being used)

    :return: Log updated with epoch statistics
    """
    model.train()
    train_loss, train_acc = 0, 0

    # loop over the training set
    for (x, y, _, v) in loader:
        (x, y, v) = (x.to(device), y.to(device), v.to(device))

        pred, _extra = model(x, v)
        loss = loss_fn(pred, y)
        opt.zero_grad()  # 0 the gradient
        loss.backward()  # backprop
        opt.step()  # update weights

        # loss & accuracy logging
        train_loss += loss
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
    if lrs is not None:
        lrs.step()
    if verbose:  # not recommended; purely debugging
        logging.info(f"LR, {lrs.get_lr()}")
        logging.info(f"train_loss {train_loss.item()}, train_acc {train_acc}")
        # plt.plot_conv_visualizations(model.conv1, )
    log["train_loss"].append(train_loss.item())
    log["train_acc"].append(train_acc)
    return log


def _dl_validate(model, loader, device, loss_fn, log, verbose=False, log_type="val"):
    """
    Verifies model performance on a dataset. Data is assumed to be validation, however can also use the training data.

    :param model: CNN or other PyTorch NN Module to evaluate
    :param loader: DataLoader containing validation data
    :param device: Current device to run on
    :param loss_fn: Callable Loss Function; requires inputs in the form of (pred_label, act_label)
    :param log: Dict containing training/validation keys for tracking
    :param verbose: Enables function verbosity
    :param log_type: String name of the log to append; one of "train" or "val"

    :return: Log updated with validation statistics
    """
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        model.eval()
        for x, y, _, v in loader:
            x, y, v = (x.to(device), y.to(device), v.to(device))
            pred, _extra = model(x, v)
            val_loss += loss_fn(pred, y)
            val_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
    log[f"{log_type}_loss"].append(val_loss.item())
    log[f"{log_type}_acc"].append(val_acc)
    if verbose:
        logging.info(f"val_loss {val_loss.item()}, val_acc {val_acc}")
    return log


def _normalize_performance_logs(logs, t_steps, t_size, v_steps, v_size):
    """
    Normalizes performance logs for DL methods

    :param logs: Dict performance logs
    :param t_steps: Training steps
    :param t_size: Training set size
    :param v_steps: Validation steps
    :param v_size: Validation set size
    :return: Updated logs
    """
    for k1, k2 in zip(["train_loss", "val_loss"], ["train_acc", "val_acc"]):
        logs[k1] = [a / t_steps if "train" in k1 else a / v_steps for a in logs[k1]]
        logs[k2] = [a / t_size if "train" in k2 else a / v_size for a in logs[k2]]
        logging.info(f"Final {k1}\t{round(logs[k1][-1], 3)}")
        logging.info(f"Final {k2}\t\t{round(logs[k2][-1], 3)}")
    return logs
