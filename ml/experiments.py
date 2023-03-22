import os

import torch
import logging
import pandas as pd
import numpy as np
import numpy.random as nprand
import pyswarms as ps
from sklearn.metrics import silhouette_score

import plots as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch import nn
from tqdm import tqdm
import nn as nets
import transformer as trans
from models import Baselines, Clustering, TSBaselines
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.model_selection import train_test_split
from data import format_name, _col_order, run_pca, run_umap, DropletDataset, ToTensor, FloatTransform, SubdivTransform


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
    def pca_reshape(x, f, flatten=True):
        x_selected = np.array([a[:, f == 1] for a in x])
        shape = x_selected.shape
        if flatten:
            x_selected = x_selected.reshape((shape[0], shape[1] * shape[2]))
        return x_selected

    def pcac_loss(f):
        """
        PCA-Clustering loss function.

        :param f: Iterable of binary features
        :return: Loss term
        """
        x_selected = pca_reshape(X, f)
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

    if not args.load:
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

        out_name = f"{args.name}_features.npy"
        np.save(f"{out_dir}results/{out_name}", pos)
        logging.info(f"Exported feature array to {out_name}")

        # Display final plots & PCA
        x_selected = pca_reshape(X, pos, flatten=False)
        plt.plot_sample_vs_mean(x_selected, y, [0, 15, 40, 55], out_dir)  # show example images
        x_selected = pca_reshape(X, pos, flatten=True)
        x_selected, _ = run_pca(x_selected, y, args.seed, out_dir=out_dir, verbose=True)  # show PCA space
        logging.info(f"Final Inverse Silhouette: {1 / silhouette_score(x_selected, y)}")
        plt.animate_pso_swarm(optimizer, out_dir)  # show PSO animation (may or may not function...)


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
    data = DropletDataset(X_data, y_data, list(range(0, len(y_data))), transforms=[
        ToTensor(), FloatTransform(),
    ])
    misc_rates = {i: [] for _x, _y, i in data}  # Misclassification rates of test samples

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
        model = nets.CMapNN(num_classes=data.labels()[1], kernel_size=ks).to(device)
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
                x, y, i = sample
                x, y = x.to(device), y.to(device)
                pred, _d = model(x)
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
            for i, d in enumerate(data):  # Sample-wise CAMs
                plt.compute_class_activation_maps(model, d[0], d[1], cam_export_dir, str(i), device, display=True)
            for cam_metric in [np.mean, np.median, np.var]:  # Aggregated CAMs
                if cam_metric not in cams:
                    cams[cam_metric] = []
                cams[cam_metric].append(plt.compute_aggregated_cams(model, data, cam_export_dir, cam_metric, str(seed),
                                                                    device, display=True))
            # conv_trend = [v[0] for k, v in conevolution.items()]  # Evolution of convolutional filters
            # plt.animate_convolution_by_epoch(conv_trend, f=5, t=E, out_dir=conv_export_dir, fname=f"_{model_name}",
            #                                  title=f"seed:{seed}")
        if args.save:  # save model (can be expensive!)
            torch.save(model.state_dict(), f"{out_dir}models/model{index}_{seed}.pt")

    # Trans-seed plotting
    misc_rates = {k: sum(v)/len(v) if len(v) > 0 else 0. for k, v in misc_rates.items()}  # Mean misclassification rates
    plt.plot_samplewise_misclassification_rates(misc_rates, 20, y_data, arguments=None, out_dir=f"{out_dir}/figs/", mname="full_conv")

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
    for (x, y, _) in loader:
        (x, y) = (x.to(device), y.to(device))

        pred, _extra = model(x)
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
        for x, y, _ in loader:
            x, y = (x.to(device), y.to(device))
            pred, _extra = model(x)
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
