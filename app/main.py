import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import os


path = os.path.dirname(os.path.abspath(__file__)) + "/"
app = FastAPI()
model = pickle.load(open(path + "../exploration/model.p","rb"))
features = pickle.load(open(path + "../exploration/features.p","rb"))


class Data(BaseModel):
    ContractTime_contract= False
    ContractTime_permanent= False
    Category_Accounting__Finance_Jobs= False
    Category_Admin_Jobs= False
    Category_Charity__Voluntary_Jobs= False
    Category_Consultancy_Jobs= False
    Category_Creative__Design_Jobs= False
    Category_Customer_Services_Jobs= False
    Category_Domestic_help__Cleaning_Jobs= False
    Category_Energy__Oil__Gas_Jobs= False
    Category_Engineering_Jobs= False
    Category_Graduate_Jobs= False
    Category_HR__Recruitment_Jobs= False
    Category_Healthcare__Nursing_Jobs= False
    Category_Hospitality__Catering_Jobs= False
    Category_IT_Jobs= False
    Category_Legal_Jobs= False
    Category_Logistics__Warehouse_Jobs= False
    Category_Maintenance_Jobs= False
    Category_Manufacturing_Jobs= False
    Category_Other_General_Jobs= False
    Category_PR__Advertising__Marketing_Jobs= False
    Category_Part_time_Jobs= False
    Category_Property_Jobs= False
    Category_Retail_Jobs= False
    Category_Sales_Jobs= False
    Category_Scientific__QA_Jobs= False
    Category_Social_work_Jobs= False
    Category_Teaching_Jobs= False
    Category_Trade__Construction_Jobs= False
    Category_Travel_Jobs= False


@app.post("/predict")
def predict(data: Data):

    #print("hello world")

    data = data.dict()

    #print(data)

    to_predict = np.array([data[f] for f in features]).reshape(1,-1)
    pred = model.predict(to_predict)

    return {"prediction" : pred.item()}
