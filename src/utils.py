import os
import pandas as pd
import mlflow
import shutil
import matplotlib.pyplot as plt


def log_pickle_artifact(x,filename, artifact_path=None):
    path = ".tmp_artifacts/"
    if not os.path.isdir(path):
        os.makedirs(path)
    # tmp save
    pd.to_pickle(x,path + filename)
    # log
    mlflow.log_artifact(path + filename,artifact_path)
    # remove
    shutil.rmtree(path)


def log_image_artifact(filename, artifact_path=None):
    path = ".tmp_artifacts/"
    if not os.path.isdir(path):
        os.makedirs(path)
    # tmp save
    plt.tight_layout()
    plt.savefig(path + filename)
    # log
    mlflow.log_artifact(path + filename,artifact_path)
    # remove
    shutil.rmtree(path)