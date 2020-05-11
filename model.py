import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.offline as py
import warnings
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pickle

df=pd.read_csv(r"D:\Users\abhisv\Desktop\Historical Product Demand.csv")
df=df.sample(10000)
df.dropna(axis=0,how='any',inplace=True)
df.reset_index(inplace=True,drop=True)

df['Date']=df['Date'].apply(pd.to_datetime)
df['year']=df['Date'].dt.year
df['month']=df['Date'].dt.month
df['Day']=df['Date'].dt.day

df.drop(['Product_Code','Product_Category','Date'],axis=1,inplace=True)


df['Order_Demand']=df['Order_Demand'].map(lambda x: x.rstrip(')'))
df['Order_Demand']=df['Order_Demand'].map(lambda x: x.lstrip('('))

print(df.dtypes)

df['Order_Demand']=df['Order_Demand'].apply(pd.to_numeric)
df['Order_Demand']=df['Order_Demand'].astype('float')
print(df.dtypes)

df_od=df.iloc[:,1]

df_od=pd.DataFrame(df_od)

def outliertreatment(df_od):
    for i in df_od:
        Q1=np.quantile(df_od,0.25)
        Q3=np.quantile(df_od,0.75)
        IQR=Q3-Q1
        LTV=Q1-1.5*IQR
        UTV=Q3+1.5*IQR
        x=np.array(df_od[i])
        p=[]
        for j in x:
            if j<LTV or j>UTV:
                p.append(df_od[i].median())
            else:
                p.append(j)
        df_od[i]=p
    
outliertreatment(df_od)
df.drop('Order_Demand',axis=1,inplace=True)

df=pd.concat([df,df_od],axis=1)

warehouse_dummy=pd.get_dummies(df['Warehouse'],drop_first=True)

df.drop('Warehouse',axis=1,inplace=True)

df=pd.concat([df,warehouse_dummy],axis=1)

y=df.iloc[:,3]
df.drop('Order_Demand',axis=1,inplace=True)
x=df

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=40)

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)

y_pred=dtr.predict(x_train)

from sklearn.metrics import r2_score
score=r2_score(y_train,y_pred)
print(score)

# Saving model to disk
pickle.dump(dtr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2015,12,2,0,0,1]]))
