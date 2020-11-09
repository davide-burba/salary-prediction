import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from copy import deepcopy
import mlflow
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import torch
from torch import nn
import mlflow
import pandas as pd
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
            y_pred = self.predict(X_valid)
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
        

class Baseline(BaseModel):
    def fit(self,X,y):
        self.median = np.median(y)
    def predict(self,X):
        return np.repeat(self.median,len(X))
    def cross_validate(self,X,y):
        data = lgb.Dataset(X,y,free_raw_data=False)
        scores = lgb.cv(self.params,data,stratified=False)
        return scores


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
            lgb.plot_importance(bst)
            log_image_artifact("feat_importance-{}.png".format(fold),"feat_importance")


class ExpActivation(nn.Module):
    def forward(self,x):
        return torch.exp(x)


class ProbNN(BaseModel):
    def __init__(self,
                lr=0.01,
                epochs = 256,
                batch_size = 256,
                num_nodes = [64],
                batch_norm = True,
                activation = torch.nn.ReLU,
                output_activation = ExpActivation,
                dropout = 0.2,
                clip_grad = 100,
                distr="weibull",
                random_state=None,
                ):
        if distr == "weibull":
            self.criterion = LossWeibull()
            self.output_features = 2
        elif distr == "loglogistic":
            self.criterion = LossLogLogistic()
            self.output_features = 2
        else:
            raise NotImplementedError()
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_nodes = num_nodes 
        self.batch_norm = batch_norm
        self.activation = activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.clip_grad = clip_grad
        self.distr = distr
        self.loss = []
        self.distr = distr
        self.random_state = random_state

        self.counter = 0
        
    def fit(self,X,y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        # prepare network
        input_features = X.shape[1]
        self.engine = NNArch(input_features,
                            self.output_features,
                            self.num_nodes,
                            self.dropout,
                            self.batch_norm,
                            self.activation,
                            self.output_activation)

        self.engine.train()
        optimizer = torch.optim.Adam(self.engine.parameters(),lr=self.lr)
        
        # prepare data ---------------------------------------------------------- TMP!!!!
        for c in X.columns:
            if X[c].dtype.name == "category":
                X[c] = X[c].cat.codes
        X = X.fillna(-1)
        x_train = torch.Tensor(X.values)
        y_train = torch.Tensor(y / 10000)

        dataloader = torch.utils.data.DataLoader(TensorLoader(x_train, y_train),
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 drop_last=False)
        self.counter += 1
        # train
        for epoch in range(self.epochs):
            epoch_loss = []
            for x,y in dataloader:
                out = self.engine(x)
                loss = self.criterion(out,y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.engine.parameters(),self.clip_grad)
                optimizer.step()
                epoch_loss.append(loss.item())
            self.loss.append(np.mean(epoch_loss))
            mlflow.log_metric("loss_{}".format(self.counter),np.mean(epoch_loss),epoch)
            
    def predict(self,X):
        self.engine.eval()
        # prepare data ---------------------------------------------------------- TMP!!!!
        for c in X.columns:
            if X[c].dtype.name == "category":
                X[c] = X[c].cat.codes
        X = X.fillna(-1)
        x = torch.Tensor(X.values)
        # forward
        out = self.engine(x)
        
        if self.distr == "weibull":
            lambda_, p_ = out[:,0],out[:,1]
            lambda_ = lambda_.detach().numpy()
            p_ = p_.detach().numpy()
            pred = lambda_ * np.log(2) * p_
        elif self.distr == "loglogistic":
            alpha_, beta_ = out[:,0],out[:,1]
            alpha_ = alpha_.detach().numpy()
            beta_ = beta_.detach().numpy()
            pred = alpha_
        else:
            raise NotImplementedError()

        return pred * 1000

    def predict_parameters(self,X):
        self.engine.eval()
        x = torch.Tensor(X.values)
        out = self.engine(x)
        
        if self.distr == "weibull":
            lambda_, p_ = out[:,0],out[:,1]
            lambda_ = lambda_.detach().numpy()
            p_ = p_.detach().numpy()
            return lambda_,p_
        elif self.distr == "loglogistic":
            alpha_, beta_ = out[:,0],out[:,1]
            alpha_ = alpha_.detach().numpy()
            beta_ = beta_.detach().numpy()
            return alpha_, beta_
        else:
            raise NotImplementedError()

            
class TensorLoader:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index],self.y[index]


class NNArch(nn.Module):
    def __init__(self, 
                 in_features,
                 out_features,
                 hidden = [64],
                 dropout = 0.2,
                 batch_norm = True,
                 activation = torch.nn.ReLU,
                 output_activation = ExpActivation,
                ):
        super().__init__()
        layers = []
        previous = in_features
        for h in hidden:
            layers.append(nn.Linear(previous,h))
            if batch_norm: layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            previous = h
        layers.append(nn.Linear(previous,out_features))
        if output_activation is not None:
            layers.append(output_activation())
        self.surv_net = nn.Sequential(*layers)

    def forward(self, input):
        out = self.surv_net(input)
        return out





class LossWeibull(nn.Module):
    """Negative Weibull log-likelyhood. 
    - lambda = model_output[:,0] 
    - p = model_output[:,1]
    """
    def __call__(self, model_output, y):
        lambda_, p_ = model_output[:,0],model_output[:,1]
        loglik = torch.log(lambda_ * p_) + (p_ - 1) * torch.log(lambda_ * y) - (lambda_ * y) ** p_
        loss = - loglik.mean()
        return loss


class LossLogLogistic(nn.Module):
    """Negative LogLogistic  log-likelyhood. 
    - alpha = model_output[:,0] 
    - beta = model_output[:,1] 
    """
    def __call__(self, model_output, y):
        alpha_, beta_ = model_output[:,0],model_output[:,1]
        loglik = torch.log(beta_ / alpha_) + (beta_ - 1) * torch.log(y / alpha_) - 2 * torch.log(1 + (y / alpha_) ** beta_)
        loss = - loglik.sum()
        return loss

