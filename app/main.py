import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os


class ModelByRegion:
    def __init__(self,bst,scale = 34122.5775755):
        self.bst = bst
        self.scale = scale
        # from https://www.statista.com/statistics/708972/average-annual-nominal-wages-of-employees-italy-by-region/
        self.salary_by_region = dict(
            Lombardia=31446,
            Trentino=30786,
            Lazio=30496,
            Emilia=30273,
            Liguria=30190,
            Piemonte=29556,
            Veneto=29286,
            Friuli=29222,
            Aosta=29202,
            Toscana=28513,
            Marche=27411,
            Abruzzo=27039,
            Campania=26904,
            Umbria=26737,
            Puglia=26410,
            Molise=26263,
            Sicilia=26133,
            Sardegna=26042,
            Calabria=25079,
            Basilicata=24308,
        )
        
    def predict(self,contract_time,category,region):
        x = pd.DataFrame(dict(contract_time=[contract_time], category=[category]),dtype="category")
        return (self.bst.predict(x) / self.scale) * self.salary_by_region[region]

    
class Data(BaseModel):
    contract_time = "permanent"
    category = "Accounting & Finance Jobs"
    region = "Lombardia"

    
path = os.path.dirname(os.path.abspath(__file__)) + "/"
app = FastAPI()
features = pickle.load(open(path + "../models/features.p","rb"))
bst = pickle.load(open(path + "../models/bst.p","rb"))
model = ModelByRegion(bst)


@app.post("/predict")
def predict(data: Data):
    
    data = data.dict()

    to_predict = [data[f] for f in features]
    pred = model.predict(*to_predict)

    return {"prediction" : pred.item()}
