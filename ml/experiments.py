import os
import torch
import logging
import pandas as pd
import numpy as np
import numpy.random as nprand
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

        state_data = X.copy()
        if args.features_selection == "top":  # since Percentile is deterministic no need to run @ each seed
            selector = SelectPercentile(score_func=mutual_info_classif, percentile=20)
            selector.fit(state_data, y)
            state_data = pd.DataFrame(selector.transform(state_data))

        # obtain model results over n seeds
        r, cm, p, dt_top = [], [], [], []  # results, confusion matricies, feature importance, decision tree splits
        for state in nprand.randint(0, 99999, size=args.num_states):
            if args.features_selection == "pca":  # PCA can be deterministic under randomizer solver; may occur in BD
                state_data = run_pca(state_data, y, state)

            train_d, test_d, train_l, test_l = train_test_split(state_data, y, test_size=0.3, stratify=y)
            baselines.data(train_d, train_l, test_d, test_l)
            result, c, _imp, _split = model(random_state=state,
                                            verbose=args.verbose,
                                            importance=args.importance,
                                            feature_names=_col_order(args.type),
                                            only_acc=args.only_acc)
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
        save_dir = "{log}{mod}/".format(log=out_dir, mod=model_name)

        # save performance results to csv file(s) with any produced plots
        if args.save:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            csv_name = format_name(args, save_dir, ".csv")
            results.to_csv(csv_name)

            if args.verbose:  # aggregate confusion matrix & misclassification rate figures & save to output directory
                plt.plot_samplewise_misclassification_rates(baselines, 22, y, args, save_dir, model_name)
                plt.plot_confusion_matrix(cm, y, args, save_dir, model_name)

            # format & save feature importances into a logging subdirectory
            if args.importance and not args.only_acc and model_name in ["logreg", "dt"]:
                importance_dir = "{sd}importance/".format(sd=save_dir)
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
                priority.to_csv(imp_name)  # need to update save dir...

                # Also save decision tree splits for analysis
                if model_name == "dt":
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
    nprand.seed(0)

    for model in models.m[args.model]:
        r, cm = [], []  # results, confusion matricies, feature importance, decision tree splits
        for s in nprand.randint(0, 99999, size=20):
            X_copy = X.copy()
            train_d, test_d, train_l, test_l = train_test_split(X_copy, y, test_size=0.3, stratify=y)
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
        results.to_csv(f"{args.model}{args.name}.csv")
        logging.info(results)

        # display & save confusion matrix
        if args.verbose:
            plt.plot_confusion_matrix(cm, y, args, out_dir, mname=args.name)


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
        X = run_pca(X, y, args.seed)

    clusters.data(X, y)
    model = clusters.m[args.model][0]
    model = model()
    clusters.plot_dendrogram(model, out_dir)


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
    cams = {"variance": [], "mean": [], "median": []}  # Track aggregated CAMs across multiple seeds

    # Reformat data & define DataSet
    X_data = np.array([np.array([np.rot90(x, k=3)]) for x in X])  # rotate so we have (time, pos) for (H, W)
    y_data = np.array(y)
    data = DropletDataset(X_data, y_data, transforms=[
        ToTensor(), FloatTransform(),
    ])
    tr_size, val_size = int(data.__len__() * spl[0]) + 1, int(data.__len__() * spl[1])

    for index, seed in enumerate(nprand.randint(0, 999999999, size=args.num_states)):
        # Sub-seed setup
        seed = int(seed)  # why????
        nprand.seed(seed)
        torch.manual_seed(seed)
        logging.critical(f"Set Seed {index+1}, {seed}")

        # Initialize training & test DataLoaders
        (trainData, testData) = random_split(data, [tr_size, val_size], generator=torch.Generator().manual_seed(seed))
        trainLoader = DataLoader(trainData, batch_size=bs, shuffle=True)
        testLoader = DataLoader(testData, batch_size=bs, shuffle=True)
        trainSteps, valSteps = len(trainLoader.dataset) // bs, len(testLoader.dataset) // bs

        # Initialize model, optimizer, logs
        ks = 3 if args.type == "processed" else 5  # adaptive kernel size
        model = nets.CMapNN(num_classes=data.labels()[1], kernel_size=ks).to(device)
        logging.info(f"Model Size: {nets.count_params(model)}")

        optimizer = Adam(model.parameters(), lr=lr)
        loss_fn = nn.NLLLoss()
        performance_log = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        conevolution = {}  # convolution + evolution lol

        if not args.load:  # Training & validation
            for e in (bar := tqdm(range(0, E))):
                performance_log = _dl_train_epoch(model, trainLoader, device, loss_fn, optimizer, performance_log)
                performance_log = _dl_validate(model, testLoader, device, loss_fn, performance_log)
                t, v = performance_log["train_acc"][e], performance_log["val_acc"][e]
                bar.set_description(f"t{t} v{v}")  # Update bar description
                if args.verbose:
                    conevolution[e] = [np.copy(model.conv3.weight.detach().numpy())]
            performance_log = _normalize_performance_logs(performance_log, trainSteps, tr_size, valSteps, val_size)
            performances["train_acc"].append(round(performance_log["train_acc"][-1], 3))
            performances["val_acc"].append(round(performance_log["val_acc"][-1], 3))
        else:  # Load from checkpoint & obtain final performances
            logging.critical(f"Loading model {index} with seed {seed}.")
            model.load_state_dict(torch.load(f"{out_dir}models/model{index}_{seed}.pt"))
            for metric in performance_log.keys():
                performance_log[metric].append(_dl_validate(model, trainLoader if "train" in metric else testLoader,
                                                            device, loss_fn, performance_log)[metric.replace("train", "val")][0])
            performance_log = _normalize_performance_logs(performance_log, trainSteps, tr_size, valSteps, val_size)  # TODO: investigate why validation logs have more than 1 value
            performances["train_acc"].append(round(performance_log["train_acc"][-1], 3))
            performances["val_acc"].append(round(performance_log["val_acc"][-1], 3))

        # Verbosity-enabled plotting
        if args.verbose:
            pass
            # plt.plot_epoch_performance(E, performance_log.keys(), out_dir, f"{index}_{seed}", *[i[1] for i in performance_log.items()])
            # for i, d in enumerate(data):  # CAMs  # TODO: export cams to seed subdir
            #     plt.compute_class_activation_maps(model, d[0], d[1], out_dir, str(i), display=True)
            #     cams["mean"].append(plt.compute_aggregated_cams(model, data, out_dir, np.mean, str(seed)))
            #     cams["median"].append(plt.compute_aggregated_cams(model, data, out_dir, np.median, str(seed)))
            #     cams["variance"].append(plt.compute_aggregated_cams(model, data, out_dir, np.var, str(seed)))
            # # Plot evolution of convolutional filters
            # conv1_trend = [v[0] for k, v in conevolution.items()]  # TODO: figure out why conv exporting errors
            # plt.animate_convolution_by_epoch(conv1_trend, f=5, t=E, out_dir=out_dir, title=f"seed: {seed}", fname=f"_{seed}")
        # Save model (expensive!)
        if args.save:
            torch.save(model.state_dict(), f"{out_dir}models/model{index}_{seed}.pt")

    # Trans-seed plotting
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
                bar.set_description(f"t{t} v{v}")  # Update bar description
            performance_log = _normalize_performance_logs(performance_log, trainSteps, tr_size, valSteps, val_size)
            performances["train_acc"].append(round(performance_log["train_acc"][-1], 3))
            performances["val_acc"].append(round(performance_log["val_acc"][-1], 3))
        else:
            logging.critical(f"Loading model {index} with seed {seed}.")
            model.load_state_dict(torch.load(f"{out_dir}models/model{index}_{seed}.pt"))
            model.eval()

        # Verbosity-enabled plotting
        if args.verbose:
            plt.plot_epoch_performance(E, performance_log.keys(), out_dir, f"{index}_{seed}", *[i[1] for i in performance_log.items()])
            plt.plot_attention_by_class(model, data, n_blocks, out_dir)  # TODO: probably needs filename arg

        # Save model (expensive!)
        if args.save:
            torch.save(model.state_dict(), f"{out_dir}models/model{index}_{seed}.pt")

    plt.plot_training_validation_heatmap(performances["train_acc"], performances["val_acc"], out_dir, tr_size, val_size)
    logging.info(f"Final results on {args.num_states} seeds: {performances}")


def _dl_train_epoch(model, loader, device, loss_fn, opt, log, verbose=False, lrs=None):
    """
    Performs one training epoch for the provided NN model.

    :param model: CNN or other PyTorch NN Module to train
    :param loader: DataLoader containing training data
    :param device: Current device to run on
    :param loss_fn: Callable Loss Function; requires inputs in the form of (pred_label, act_label)
    :param opt: Model Optimizer
    :param log: Dict containing training/validation keys for tracking
    :param verbose: Enables

    :return: Log updated with epoch statistics
    """
    model.train()
    train_loss, train_acc = 0, 0

    # loop over the training set
    for (x, y) in loader:
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
        logging.info(f"LR, {lrs.get_lr()}")

    if verbose:  # not recommended; purely debugging
        plt.plot_conv_visualizations(model.conv1, )
    log["train_loss"].append(train_loss.item())
    log["train_acc"].append(train_acc)
    # logging.info(f"train_loss {train_loss.item()}, train_acc {train_acc}")
    return log


def _dl_validate(model, loader, device, loss_fn, log, verbose=False):
    """
    Verifies model performance on the validation set

    :param model: CNN or other PyTorch NN Module to evaluate
    :param loader: DataLoader containing validation data
    :param device: Current device to run on
    :param loss_fn: Callable Loss Function; requires inputs in the form of (pred_label, act_label)
    :param log: Dict containing training/validation keys for tracking
    :param verbose: Enables function verbosity

    :return: Log updated with validation statistics
    """
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        model.eval()
        for x, y in loader:
            x, y = (x.to(device), y.to(device))
            pred, _extra = model(x)
            val_loss += loss_fn(pred, y)
            val_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
    log["val_loss"].append(val_loss.item())
    log["val_acc"].append(val_acc)
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
