import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from copy import deepcopy
import mlflow
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
# local
from utils import log_image_artifact


class BaseModel:
    def fit(self,X,y):
        pass
    def predict(self,X):
        pass
    def cross_validate(self,X,y,n_splits=5,random_state=None):

        mape,mae = [],[]
        folds = KFold(n_splits,shuffle=True, random_state=random_state)
        for idx_train,idx_valid in folds.split(X,y):
            # split
            X_train,X_valid = X.loc[idx_train],X.loc[idx_valid]
            y_train,y_valid = y[idx_train],y[idx_valid]
            # fit
            self.fit(X_train,y_train)
            # predict
            y_pred = self.predict(X_valid,y_valid)
            # evaluate
            mae.append(np.abs(y_pred - y_valid).mean())
            mape.append(np.abs((y_pred - y_valid) / y_valid).mean())
        
        scores = dict(
            mape_mean = np.mean(mape),
            mae_mean = np.mean(mae),
            mape_std = np.std(mape),
            mae_std = np.std(mae))

        artifacts = dict()
        return scores, artifacts

    def log_artifacts(self,artifacts):
        pass
        
        
class LightGBM(BaseModel):
    def __init__(self,params):
        self.params = params

    def fit(self,X,y):
        data = lgb.Dataset(X,y,free_raw_data=False)
        self.engine = lgb.train(self.params,data)

    def predict(self,X):
        ypred = self.engine.predict(data)
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
            #fig=plt.figure()
            
            #fig=plt.figure()
            #ax=plt.subplot()
            #plt.close()
            #plt.plot([1,2,3,4])
            lgb.plot_importance(bst)
            #plt.close(fig)
            log_image_artifact("feat_importance-{}.png".format(fold),"feat_importance")


class Baseline(BaseModel):
    def fit(self,X,y):
        self.median = np.median(y)
    def predict(self,X):
        return np.repeat(self.median,len(X))
    def cross_validate(self,X,y):
        data = lgb.Dataset(X,y,free_raw_data=False)
        scores = lgb.cv(self.params,data,stratified=False)
        return scores