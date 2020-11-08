import os
import pandas as pd
import mlflow
import shutil

def log_pickle_artifact(x,filename):
    path = ".tmp_artifacts/"
    if not os.path.isdir(path):
        os.makedirs(path)
    # tmp save
    pd.to_pickle(x,path + filename)
    # log
    mlflow.log_artifact(path + filename)
    # remove
    shutil.rmtree(path)