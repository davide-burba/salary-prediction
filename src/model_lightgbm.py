import pandas as pd
import numpy as np
import lightgbm as lgb

try:
    from sklearn.model_selection import KFold
    from copy import deepcopy
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use('Agg')
    import mlflow
    # local
    from utils import log_image_artifact
except:
    Warning("Inference mode")

from model_base import BaseModel


class LightGBM(BaseModel):
    def __init__(self,params):
        self.params = params

    def fit(self,X,y,X_valid=None,y_valid=None):
        data = lgb.Dataset(X,y,free_raw_data=False)
        self.engine = lgb.train(self.params,data)

    def predict(self,X):
        ypred = self.engine.predict(X)
        return ypred

    def cross_validate(self,X,y,n_splits=5,random_state=None):
        folds = KFold(n_splits,shuffle=True, random_state=random_state)
        data = lgb.Dataset(X,y,free_raw_data=False)
        artifacts = lgb.cv(
            self.params,
            data,
            folds = folds,
            metrics = ["mape","mae"],
            return_cvbooster = True)

        scores = dict(
            mape = np.mean(artifacts['mape-mean']),
            mae = np.mean(artifacts['l1-mean']),
            mape_std = np.std(artifacts['mape-stdv']),
            mae_std = np.std(artifacts['l1-stdv']))

        return scores,artifacts

    def log_artifacts(self,artifacts):
        for k in ['mape-mean', 'l1-mean', 'mape-stdv', 'l1-stdv']:
            for step,v in enumerate(artifacts[k]):
                mlflow.log_metric("boosting_{}".format(k),v,step)

        all_bst = artifacts["cvbooster"].boosters

        for fold,bst in enumerate(all_bst):
            plt.ioff()
            lgb.plot_importance(bst)
            log_image_artifact("feat_importance-{}.png".format(fold),"feat_importance")
