import pandas as pd

class DataLoader:
    def __init__(self,root,features = None, alpha = .01):
        self.col_y = 'reddito_netto'
        self.col_utils = ['n_comp_famiglia','n_percett_famiglia','n_lavor_famiglia',]
        self.col_sens_feat = ['sex','cit_italiana']
        self.col_feat = [
            'partime',
            'contratto',
            'dimensioni_azienda',
            'ore_settimana',
            'titolo_studio',
            'tipo_laurea',
            'selode',
            'annoedu',
            'tipo_diploma',
            'qualifica',
            'settore',
            'regione',
            'ETA',
            'codice_lavoro_isco',
            'eta_discreta_5',
            'status_lavoratore',
            'settore_less',
            'zona_less',
            'zona',
            'ampiezza_comune_less',
            'ampiezza_comune',
            'voto_perc',
            "n_esp_lavorative",
            "anni_contributi", 
            "eta_primo_lavoro",
            "eta_lavoro_corrente",
            ]

        self.col_all = [self.col_y] + self.col_utils + self.col_sens_feat + self.col_feat
        self.col_num = ['reddito_netto','ore_settimana','annoedu','ETA','voto_perc',
                        "n_esp_lavorative","anni_contributi", "eta_primo_lavoro","eta_lavoro_corrente",]
        self.col_cat = [v for v in self.col_all if v not in self.col_num]
        
        if features is None:
            features =  self.col_sens_feat +  self.col_feat
        self.cat_feat = [v for v in features if v in self.col_cat]
        self.num_feat = [v for v in features if v in self.col_num]

        # load data
        self.df = pd.read_csv(root + "data/preprocessed/bancaditalia/2016.csv")

        # process data
        self._add_time_distances()

        # drop extrema
        if alpha > 0:
            low = self.df[self.col_y].quantile(alpha / 2)
            high = self.df[self.col_y].quantile(1 - alpha / 2)
            self.df = self.df[(low < self.df[self.col_y]) & (self.df[self.col_y] < high)]
            self.df = self.df.reset_index(drop=True)

        self.X = self.df[features].copy()
        self.X[self.cat_feat] = self.X[self.cat_feat].astype("category")
        self.y = self.df[self.col_y].copy()

    def _add_time_distances(self):
        # fix time variables
        self.df['anni_da_edu'] = 2016 - self.df['annoedu']
        self.df['eta_lavoro_corrente'] = 2016 - self.df['eta_lavoro_corrente'] # check this