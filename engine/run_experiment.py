import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import yaml
import argparse
from datetime import datetime

ROOT = "../"
sys.path.insert(0,ROOT + "src/")

from data_management import DataLoader
from models import LightGBM

DEFAULT_ARGS = dict(
    path_dir = "debug/",
    path_pattern = "",
)


def cross_validate(args):
    # get data
    X,y = get_data(args)
    # get model
    model = get_model(args)
    # train
    scores = model.cross_validate(X,y)
    # show scores
    mape = scores['mape-mean'][-1]
    mae = scores['l1-mean'][-1]
    print("***** CV *****")
    print("MAPE: {}%    MAE: {} € \n".format(round(100*mape,2),round(mae,2)))

    # save
    if args["path"] is not None:
        # TODO
        pass


def train_model(args):
    # get data
    X,y = get_data(args)
    # get model
    model = get_model(args)
    # train
    model.fit(X,y)

    # save
    if args["path"] is not None:
        pd.to_pickle(model, args["path"] + "model.p")
        pd.to_pickle(list(X.columns.values), args["path"] + "features.p")


def get_model(args):
    if args["model"] == "LightGBM":
        model = LightGBM(args["model_args"])
    elif args["model"] == "Baseline":
        model = Baseline(args["model_args"])
    return model


def get_data(args):
    loader = DataLoader(
        ROOT, 
        args["data_params"]["features"], 
        alpha = args["data_params"]["alpha"])
    X = loader.X
    y = loader.y.values

    return X,y


def main(args):
    print("***** args *****")
    print(args,"\n")

    if args["action"] == "cross_validate":
        cross_validate(args)
    elif args["action"] == "train_model":
        train_model(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default = None)
    parsed = vars(parser.parse_args())

    # config file
    if parsed["config"] is not None:
        with open(parsed["config"],"r") as f:
            args = yaml.safe_load(f)
    else:
        args = dict()

    # set default values
    for k in DEFAULT_ARGS:
        if k not in args:
            args[k] = DEFAULT_ARGS[k]

    # set output directory
    if args["path_dir"] is not None:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args["path"] = ROOT + "experiments/" + args["path_dir"] + now + args["path_pattern"] + "/"
        if not os.path.isdir(args["path"]):
            os.makedirs(args["path"])
    else: 
        args["path"] = None

    # run experiment
    main(args)