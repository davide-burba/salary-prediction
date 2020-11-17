import pandas as pd
import numpy as np
import torch
from torch import nn
try:
    import lightgbm as lgb
    from sklearn.model_selection import KFold
    from copy import deepcopy
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use('Agg')
    import mlflow
except:
    Warning("torch inference mode")

# local
from utils import log_image_artifact


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
            median_ae_train = np.median(np.abs(y_pred - y_valid))
            median_ape_train = np.median(np.abs((y_pred - y_valid) / y_valid))
        scores = dict(
            mape_mean = np.mean(mape),
            mae_mean = np.mean(mae),
            median_ae_train_mean = np.mean(median_ae_train),
            median_ape_train_mean = np.mean(median_ape_train),
            mape_std = np.std(mape),
            mae_std = np.std(mae),
            median_ae_train_std = np.std(median_ae_train),
            median_ape_train_std = np.std(median_ape_train)
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


class LightGBM(BaseModel):
    def __init__(self,params):
        self.params = params

    def fit(self,X,y,X_valid=None,y_valid=None):
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
                embedding_size = 4,
                batch_norm = True,
                activation = torch.nn.ReLU,
                output_activation = torch.nn.Softplus,#ExpActivation,
                dropout = 0.2,
                clip_grad = 1000,
                distr="normal",
                y_unit = 1000,
                random_state=None,
                ):
        if distr == "normal":
            self.criterion = LossNormal()
            self.output_features = 2
        elif distr == "weibull":
            self.criterion = LossWeibull()
            self.output_features = 2
        elif distr == "loglogistic":
            self.criterion = LossLogLogistic()
            self.output_features = 2
        elif distr == "mae":
            self.criterion = nn.L1Loss()
            self.output_features = 1
        elif distr == "mape":
            self.criterion = LossMape()
            self.output_features = 1
        else:
            raise NotImplementedError()

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_nodes = num_nodes 
        self.embedding_size = embedding_size
        self.batch_norm = batch_norm
        self.activation = activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.clip_grad = clip_grad
        self.distr = distr
        self.random_state = random_state
        self.y_unit = y_unit

        self.counter = 0
        
    def fit(self,X_train,y_train,X_valid=None,y_valid=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        # prepare data 
        self.n_categorical = (X_train.dtypes == "category").sum()

        x_training = torch.Tensor(X_train.astype("float64").values)
        y_training = torch.Tensor(y_train / self.y_unit).reshape(-1,1)
        dataloader_train = torch.utils.data.DataLoader(TensorLoader(x_training, y_training),
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    drop_last=False)
        if X_valid is not None:
            x_validation = torch.Tensor(X_valid.astype("float64").values)
            y_validation = torch.Tensor(y_valid / self.y_unit).reshape(-1,1)
            dataloader_valid = torch.utils.data.DataLoader(TensorLoader(x_validation, y_validation),
                                                        batch_size=self.batch_size,
                                                        shuffle=False,
                                                        drop_last=False)
        # prepare network
        input_features = X_train.shape[1]
        self.embedding_cat_sizes = [X_train[c].cat.categories.shape[0] for c in X_train.columns[:self.n_categorical]]
        self.engine = NNArch(input_features,
                            self.output_features,
                            self.embedding_cat_sizes,
                            self.embedding_size,
                            self.num_nodes,
                            self.dropout,
                            self.batch_norm,
                            self.activation,
                            self.output_activation)

        self.engine.train()
        optimizer = torch.optim.Adam(self.engine.parameters(),lr=self.lr)
        
        self.counter += 1
        # train
        for epoch in range(self.epochs):
            epoch_loss = []
            for x,y in dataloader_train:
                out = self.engine(x)
                loss = self.criterion(out,y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.engine.parameters(),self.clip_grad)
                optimizer.step()
                epoch_loss.append(loss.item())

            batch_loss = np.mean(epoch_loss)
            y_pred = self.predict(X_train)

            mape = np.abs((y_train - y_pred) / y_train).mean()
            mae = np.abs((y_train - y_pred)).mean()

            mlflow.log_metric("loss_{}_train".format(self.counter),batch_loss,epoch)
            mlflow.log_metric("mape_{}_train".format(self.counter),mape,epoch)
            mlflow.log_metric("mae_{}_train".format(self.counter),mae,epoch)

            if X_valid is not None:
                epoch_loss = []
                for x,y in dataloader_valid:
                    out = self.engine(x)
                    loss = self.criterion(out,y)
                    epoch_loss.append(loss.item())

                batch_loss = np.mean(epoch_loss)
                y_pred = self.predict(X_valid)
                mape = np.abs((y_valid - y_pred) / y_valid).mean()
                mae = np.abs((y_valid - y_pred)).mean()
                mlflow.log_metric("loss_{}_valid".format(self.counter),batch_loss,epoch)
                mlflow.log_metric("mape_{}_valid".format(self.counter),mape,epoch)
                mlflow.log_metric("mae_{}_valid".format(self.counter),mae,epoch)
                
    def predict(self,X):

        self.engine.eval()
        x = torch.Tensor(X.astype("float64").values)

        # forward
        out = self.engine(x)
        if self.distr == "normal":
            mu_, _ = out[:,0],out[:,1]
            mu_ = mu_.detach().numpy()
            pred = mu_
        elif self.distr == "weibull":
            lambda_, p_ = out[:,0],out[:,1]
            lambda_ = lambda_.detach().numpy()
            p_ = p_.detach().numpy()
            pred = lambda_ * np.log(2) * p_
        elif self.distr == "loglogistic":
            alpha_, beta_ = out[:,0],out[:,1]
            alpha_ = alpha_.detach().numpy()
            beta_ = beta_.detach().numpy()
            pred = alpha_
        elif self.distr == "mae" or self.distr == "mape":
            pred = out[:,0].detach().numpy()
        else:
            raise NotImplementedError()

        return pred * self.y_unit

    def predict_parameters(self,X):

        self.engine.eval()
        x = torch.Tensor(X.astype("float64").values)
        out = self.engine(x)
        
        if self.distr == "normal":
            mu_, sigma_ = out[:,0],out[:,1]
            mu_ = mu_.detach().numpy()
            sigma_ = sigma_.detach().numpy()
            return mu_,sigma_
        elif self.distr == "weibull":
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
                 embedding_cat_sizes = [],
                 embedding_size = 4,
                 hidden = [64],
                 dropout = 0.2,
                 batch_norm = True,
                 activation = torch.nn.ReLU,
                 output_activation = ExpActivation,
                ):
        super().__init__()

        # embedding
        self.n_embeddings = len(embedding_cat_sizes)
        self.embedding_size = embedding_size
        self.embedding_layers = []
        for dict_size in embedding_cat_sizes:
            self.embedding_layers.append(nn.Embedding(dict_size,embedding_size))
        self.embedding_layers = nn.ModuleList(self.embedding_layers)

        # fully-connected
        layers = []
        previous = in_features + self.n_embeddings * (embedding_size - 1)
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

    def forward(self, x):
        embedded = []
        for i in range(self.n_embeddings):
            emb = self.embedding_layers[i](x[:,i].long())
            embedded.append(emb)   
        embedded = torch.cat(embedded,axis=1)

        x = torch.cat([embedded,x[:,self.n_embeddings:]],axis=1)
        out = self.surv_net(x)
        return out


class LossMape(nn.Module):
    """mape"""
    def __call__(self, model_output, y):
        loss = torch.abs((model_output - y) / y).mean()
        return loss


class LossNormal(nn.Module):
    """Negative Normal log-likelyhood. 
    - mu = model_output[:,0] 
    - sigma = model_output[:,1]
    """
    def __call__(self, model_output, y):
        mu_, sigma_ = model_output[:,0],model_output[:,1]
        loglik = - torch.log(sigma_) - 0.5 * ((y[:,0] - mu_)  / sigma_) ** 2
        loss = - loglik.mean()
        return loss


class LossWeibull(nn.Module):
    """Negative Weibull log-likelyhood. 
    - lambda = model_output[:,0] 
    - p = model_output[:,1]
    """
    def __call__(self, model_output, y):
        lambda_, p_ = model_output[:,0],model_output[:,1]
        loglik = torch.log(lambda_ * p_) + (p_ - 1) * torch.log(lambda_ * y[:,0]) - (lambda_ * y[:,0]) ** p_
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


