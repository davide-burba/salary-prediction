import argparse
import requests



parser = argparse.ArgumentParser()
parser.add_argument('--contract_time', type=str, default = "permanent")
parser.add_argument('--category', type=str, default = "Creative__Design")
args = vars(parser.parse_args())
contract_time = args["contract_time"]
category = args['category']

to_predict_dict = dict(
    ContractTime_contract=False,
    ContractTime_permanent=False,
    Category_Accounting__Finance_Jobs=False,
    Category_Admin_Jobs=False,
    Category_Charity__Voluntary_Jobs=False,
    Category_Consultancy_Jobs=False,
    Category_Creative__Design_Jobs=False,
    Category_Customer_Services_Jobs=False,
    Category_Domestic_help__Cleaning_Jobs=False,
    Category_Energy__Oil__Gas_Jobs=False,
    Category_Engineering_Jobs=False,
    Category_Graduate_Jobs=False,
    Category_HR__Recruitment_Jobs=False,
    Category_Healthcare__Nursing_Jobs=False,
    Category_Hospitality__Catering_Jobs=False,
    Category_IT_Jobs=False,
    Category_Legal_Jobs=False,
    Category_Logistics__Warehouse_Jobs=False,
    Category_Maintenance_Jobs=False,
    Category_Manufacturing_Jobs=False,
    Category_Other_General_Jobs=False,
    Category_PR__Advertising__Marketing_Jobs=False,
    Category_Part_time_Jobs=False,
    Category_Property_Jobs=False,
    Category_Retail_Jobs=False,
    Category_Sales_Jobs=False,
    Category_Scientific__QA_Jobs=False,
    Category_Social_work_Jobs=False,
    Category_Teaching_Jobs=False,
    Category_Trade__Construction_Jobs=False,
    Category_Travel_Jobs=False,
    )

to_predict_dict["ContractTime_" + contract_time] = True
to_predict_dict["Category_" + category + "_Jobs"] = True

url = 'http://127.0.0.1:8000/predict'
r = requests.post(url,json=to_predict_dict).json()
print("Salary prediction:",r["prediction"])
