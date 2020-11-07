import numpy as np
import lightgbm as lgb

class BaseModel:
    def fit(self,X,y):
        pass
    def predict(self,X):
        pass

        
class LightGBM(BaseModel):
    def __init__(self,params):
        self.params = params

    def fit(self,X,y):
        data = lgb.Dataset(X,y,free_raw_data=False)
        self.engine = lgb.train(self.params,data)

    def predict(self,X):
        pass

    def cross_validate(self,X,y):
        data = lgb.Dataset(X,y,free_raw_data=False)
        scores = lgb.cv(self.params,data,stratified=False)
        return scores


class Baseline(BaseModel):
    def fit(self,X,y):
        self.median = np.median(y)
    def predict(self,X):
        return np.repeat(self.median,len(X))