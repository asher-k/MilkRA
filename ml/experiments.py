import os
import torch
import logging
import pandas as pd
import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt
import plots as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import nn as nets
from models import Baselines, Clustering, TSModels
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.model_selection import train_test_split
from data import format_name, _col_order, run_pca, run_umap, DropletDataset, ToTensor, FloatTransform, SubdivTransform


def classify_baselines(args, X, y, logs_dir):
    """
    Classification experiment on baseline non-time series models

    :param args: Command-line ArgParser
    :param X: Droplet data
    :param y: Droplet classes
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
        save_dir = "{log}{mod}/".format(log=logs_dir, mod=model_name)

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


def classify_dl(args, X, y):
    """
    Classification experiment with DL models.

    :param args: Command-line ArgParser
    :param X: Droplet data
    :param y: Droplet classes
    """
    lr, bs, E, spl = 1e-4, 6, 5, (0.667, 0.333)  # 70-30 train-test split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    performances = {"train_acc": [], "val_acc": []}

    for index, seed in enumerate(nprand.randint(0, 99999, size=args.num_states)):
        seed = int(seed)  # why????
        nprand.seed(seed)
        torch.manual_seed(seed)
        logging.critical(f"Set Seed {index+1}, {seed}")

        # Prepare data and initialize training & test DataLoaders
        X_data = np.array([np.array([np.rot90(x, k=3)]) for x in X])  # rotate so we have (time, pos) for (H, W)
        y_data = np.array(y)
        data = DropletDataset(X_data, y_data, transforms=[
            ToTensor(), FloatTransform(), #SubdivTransform(),
        ])
        tr_size, val_size = int(data.__len__()*spl[0])+1, int(data.__len__()*spl[1])
        (trainData, testData) = random_split(data, [tr_size, val_size], generator=torch.Generator().manual_seed(seed))
        trainLoader = DataLoader(trainData, batch_size=bs, shuffle=True)
        testLoader = DataLoader(testData, batch_size=bs, shuffle=True)
        trainSteps, valSteps = len(trainLoader.dataset) // bs, len(testLoader.dataset) // bs

        # init model, optimizer, logs
        model = nets.CMapNN(num_classes=4, ks=3).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        loss_fn = nn.NLLLoss()
        performance_log = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        conevolution = {}  # convoluton + evolution

        # begin training & track convolutional filters at each epoch
        for e in range(0, E):
            logging.info(f"Epoch: {e}")
            performance_log = _train_epoch(model, trainLoader, device, loss_fn, optimizer, performance_log)
            performance_log = _validate(model, testLoader, device, loss_fn, performance_log)
            if args.verbose:
                conevolution[e] = [np.copy(model.conv3.weight.detach().numpy())]

        # compute actual performances by aggregating across batches/steps, then save for trans-seed tracking
        for k1, k2 in zip(["train_loss", "val_loss"], ["train_acc", "val_acc"]):
            performance_log[k1] = [a/trainSteps if "train" in k1 else a/valSteps for a in performance_log[k1]]
            performance_log[k2] = [a/tr_size if "train" in k2 else a/val_size for a in performance_log[k2]]
            logging.info(f"Final {k1}\t{round(performance_log[k1][-1], 3)}")
            logging.info(f"Final {k2}\t\t{round(performance_log[k2][-1], 3)}")
        performances["train_acc"].append(round(performance_log["train_acc"][-1], 3))
        performances["val_acc"].append(round(performance_log["val_acc"][-1], 3))

        # Verbosity-enabled plotting
        if args.verbose:
            plt.plot_epoch_performance(E, performance_log.keys(), *[i[1] for i in performance_log.items()])
            for i, d in enumerate(data):  # CAMs
                plt.plot_class_activation_maps(model, d[0], d[1], args.logs_dir, str(i))
            # Plot evolution of convolutional filters
            conv1_trend = [v[0] for k, v in conevolution.items()]
            logging.info(np.sum(np.abs(conv1_trend[0] - conv1_trend[-1])))
            plt.animate_convolution_by_epoch(conv1_trend, f=5, t=E, out_dir=args.logs_dir, title=f"seed{seed}",
                                             fname=f"{args.name}:{seed}_convolutions", )
    # Trans-seed plotting
    plt.plot_training_validation_performance(performances["train_acc"], performances["val_acc"])
    if args.verbose:
        pass


def classify_ts(args, X, y):
    """
    Classification experiment with Time-series models.

    :param args: Command-line ArgParser
    :param X: Droplet data
    :param y: Droplet classes
    """
    models = TSModels()
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
        plt.clf()
        cm = np.sum(cm, axis=0)
        cm = np.round(cm / np.sum(cm, axis=1), 3)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted([l[:4] for l in set(y)]))
        cm_display.plot()
        plt.savefig(f"cm_{args.model}{args.name}.png")
        plt.show()


def clustering(args, X, y):
    """
    Clustering experiment on our PCA'd data.

    :param args: Command-line ArgParser
    :param X: Droplet data
    :param y: Droplet classes
    """
    clusters = Clustering()
    if args.features_selection == "pca":  # PCA can be deterministic under randomizer solver; may occur in BD
        X = run_pca(X, y, args.seed)

    clusters.data(X, y)
    model = clusters.m[args.model][0]
    model = model()
    clusters.dendrogram(model)


def _train_epoch(model, loader, device, loss_fn, opt, log, verbose=False):
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

    if verbose:  # not recommended; purely debugging
        plt.plot_conv_visualizations(model.conv1, )
    log["train_loss"].append(train_loss.item())
    log["train_acc"].append(train_acc)
    logging.info(f"train_loss {train_loss.item()}, train_acc {train_acc}")
    return log


def _validate(model, loader, device, loss_fn, log):
    """
    Verifies model performance on the validation set

    :param model: CNN or other PyTorch NN Module to evaluate
    :param loader: DataLoader containing validation data
    :param device: Current device to run on
    :param loss_fn: Callable Loss Function; requires inputs in the form of (pred_label, act_label)
    :param log: Dict containing training/validation keys for tracking

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
    logging.info(f"val_loss {val_loss.item()}, val_acc {val_acc}")
    return log
