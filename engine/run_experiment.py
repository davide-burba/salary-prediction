import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import yaml
import argparse
from datetime import datetime
import mlflow

ROOT = os.getcwd() +  "/../"
sys.path.insert(0,ROOT + "src/")

from data_management import DataLoader
from models import LightGBM
from utils import log_pickle_artifact


DEFAULT_ARGS = dict(
    path_dir = "debug/",
    path_pattern = "",
    store_artifacts = False,
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

    # log scores
    for k in scores:
        for step,v in enumerate(scores[k]):
            mlflow.log_metric(k,v,step)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("mae",mae)


def train_model(args):
    # get data
    X,y = get_data(args)
    # get model
    model = get_model(args)
    # train
    model.fit(X,y)
    # save
    if args["store_artifacts"]:
        log_pickle_artifact(model,"model.p")
        log_pickle_artifact(list(X.columns.values),"features.p")


def get_model(args):
    if args["model"] == "LightGBM":
        model = LightGBM(args["model_args"])
    elif args["model"] == "Baseline":
        model = Baseline(args["model_args"])
    return model


def get_data(args):
    loader = DataLoader(
        ROOT, 
        args["data_args"]["features"], 
        alpha = args["data_args"]["alpha"])
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

    print(ROOT)

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

    # set logging
    mlflow.set_tracking_uri("file:" + ROOT + "mlruns/")
    mlflow.set_experiment(args["experiment"])
    mlflow.start_run(run_name=args["run_name"])
    mlflow.set_tags(args["tags"])

    # store source code and config for reproducibility
    mlflow.log_artifact(ROOT + "src")
    mlflow.log_artifact(ROOT + "engine/run_experiment.py")
    mlflow.log_artifact(parsed["config"])

    # store args (flatten nested dictionaries)
    for k in args:
        if type(args[k]) is not dict:
            mlflow.log_param(k,args[k])
        else:
            for sub_k in args[k]:
                mlflow.log_param(k + "-" + sub_k,args[k][sub_k])

    # run experiment
    main(args)