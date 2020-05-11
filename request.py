import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'year':2019, 'month':9, 'Day':6,'Whse_C':1,'Whse_J':0,'Whse_S':0})

print(r.json())