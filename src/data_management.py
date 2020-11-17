import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


Y_COL = 'reddito_netto'
ALL_FEATURES = [
    'partime', 'contratto', 'dimensioni_azienda','ore_settimana', 
    'sex', 'cit_italiana', 'titolo_studio', 'tipo_laurea',
    'selode', 'anni_da_edu', 'tipo_diploma', 'qualifica', 'settore', 'regione',
    'ETA', 'codice_lavoro_isco', 'eta_discreta_5', 'status_lavoratore',
    'settore_less', 'zona_less', 'zona', 'ampiezza_comune_less',
    'ampiezza_comune', 'n_esp_lavorative', 'anni_contributi',
    'anni_da_primo_lavoro', 'anni_da_lavoro_corrente', 'voto_perc']
NUMERIC = [
    'reddito_netto','ore_settimana','anni_da_edu','ETA','voto_perc',
    "anni_da_primo_lavoro", "anni_da_lavoro_corrente"]
CATEGORICAL = [v for v in ALL_FEATURES if v not in NUMERIC]


class DataLoader:
    def __init__(self,root,features = None, alpha = .01):
        
        # load data
        self.df = pd.read_csv(root + "data/preprocessed/bancaditalia/2016.csv")

        # process data
        self._add_time_distances()
        self._drop_extrema(alpha)
        self._fillna()

        # set features
        if features is None:
        	features = ALL_FEATURES
        self.cat_feat = [v for v in features if v in CATEGORICAL]
        self.num_feat = [v for v in features if v in NUMERIC]
        # sanity check
        assert len(features) == len(self.cat_feat) + len(self.num_feat)

        # put cat_feat at the beginning
        features = self.cat_feat + self.num_feat

       	# set X,y
        self.X = self.df[features].copy()
        self.X[self.cat_feat] = self.X[self.cat_feat].astype("category")
        self.y = self.df[Y_COL].copy()

    def _add_time_distances(self):
        """fix time variables"""
        self.df['anni_da_edu'] = 2016 - self.df['annoedu']
        self.df['anni_da_lavoro_corrente'] = 2016 - self.df['eta_lavoro_corrente']
        self.df['anni_da_primo_lavoro'] = 2015 - self.df['eta_primo_lavoro']

    def _drop_extrema(self,alpha):
        """drop extrema in y column"""
        self.alpha = alpha
        if alpha > 0:
            low = self.df[Y_COL].quantile(alpha / 2)
            high = self.df[Y_COL].quantile(1 - alpha / 2)
            self.df = self.df[(low < self.df[Y_COL]) & (self.df[Y_COL] < high)]
            self.df = self.df.reset_index(drop=True)

    def _fillna(self):
        """make Nans a category for categorical, else -1"""
        for c in self.df.columns:
            if self.df[c].dtype.name is "category":
                if self.df[c].isnull().sum() > 0:
                    self.df[c].cat.add_categories("Nan",inplace=True)
                    self.df[c] = self.df[c].fillna("Nan")
            else:
                self.df[c] = self.df[c].fillna(-1)


class CatEncoder:
    def fit_transform(self,X): 
        self.encoder = OrdinalEncoder()
        self.n_categorical = (X.dtypes == "category").sum()
        self.encoder.fit(X[X.columns[:self.n_categorical]])
        return self.transform(X)

    def transform(self,X):
        x_cat = self.encoder.transform(X[X.columns[:self.n_categorical]])
        X[X.columns[:self.n_categorical]] = x_cat
        X[X.columns[:self.n_categorical]] = X[X.columns[:self.n_categorical]].astype("category")
        return X