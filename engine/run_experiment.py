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
import logging

ROOT = os.getcwd() +  "/../"
sys.path.insert(0,ROOT + "src/")

from data_management import DataLoader
from models import LightGBM
from utils import log_pickle_artifact


DEFAULT_ARGS = dict(
    path_dir = "debug/",
    path_pattern = "",
    store_artifacts = False,
    random_state = 1234,
    cv_state = 1234,
    n_splits=5,
)

def cross_validate(args):
    # get data
    X,y = get_data(args)
    # get model
    model = get_model(args)
    # train
    scores,artifacts = model.cross_validate(X,y,args["n_splits"],args["cv_state"])
    # show scores
    logging.info("***** CV scores *****")
    logging.info(scores)
    # log scores
    for k in scores: 
        mlflow.log_metric(k,scores[k])
    # log artifacts
    if args["store_artifacts"]:
        model.log_artifacts(artifacts)
        


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

    # fix randomness
    np.random.seed(args["random_state"])

    # choose action
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

    # set default args values
    for k in DEFAULT_ARGS:
        if k not in args:
            args[k] = DEFAULT_ARGS[k]

    # set mlflow logging
    mlflow.set_tracking_uri("file:" + ROOT + "mlruns/")
    mlflow.set_experiment(args["experiment"])
    mlflow.start_run(run_name=args["run_name"])
    mlflow.set_tags(args["tags"])

    # store source code and config for reproducibility
    mlflow.log_artifact(ROOT + "src","reproduce")
    mlflow.log_artifact(ROOT + "engine/run_experiment.py","reproduce/engine")
    mlflow.log_artifact(parsed["config"],"reproduce/engine")

    # store args (flatten nested dictionaries)
    for k in args:
        if type(args[k]) is not dict:
            mlflow.log_param(k,args[k])
        else:
            for sub_k in args[k]:
                mlflow.log_param(k + "-" + sub_k,args[k][sub_k])

    # run experiment
    main(args)