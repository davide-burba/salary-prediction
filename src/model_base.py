
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from copy import deepcopy
    


class BaseModel:
    def fit(self,X,y,X_valid=None,y_valid=None):
        pass
    def predict(self,X):
        pass
    def cross_validate(self,X,y,n_splits=5,random_state=None):

        mape,mae = [],[]
        median_ape,median_ae = [],[]
        folds = KFold(n_splits,shuffle=True, random_state=random_state)
        for idx_train,idx_valid in folds.split(X,y):
            # split
            X_train,X_valid = X.loc[idx_train],X.loc[idx_valid]
            y_train,y_valid = y[idx_train],y[idx_valid]
            # fit
            self.fit(X_train,y_train,
                     X_valid,y_valid) # for monitoring
            # predict
            y_pred = self.predict(X_valid)
            # evaluate
            mae.append(np.abs(y_pred - y_valid).mean())
            mape.append(np.abs((y_pred - y_valid) / y_valid).mean())
            median_ae = np.median(np.abs(y_pred - y_valid))
            median_ape = np.median(np.abs((y_pred - y_valid) / y_valid))
        scores = dict(
            mape_mean = np.mean(mape),
            mae_mean = np.mean(mae),
            median_ae_mean = np.mean(median_ae),
            median_ape_mean = np.mean(median_ape),
            mape_std = np.std(mape),
            mae_std = np.std(mae),
            median_ae_std = np.std(median_ae),
            median_ape_std = np.std(median_ape)
            )

        artifacts = dict()
        return scores, artifacts

    def log_artifacts(self,artifacts):
        pass
        

class Baseline(BaseModel):
    def fit(self,X,y):
        self.median = np.median(y)
    def predict(self,X):
        return np.repeat(self.median,len(X))
    def cross_validate(self,X,y):
        data = lgb.Dataset(X,y,free_raw_data=False)
        scores = lgb.cv(self.params,data,stratified=False)
        return scores