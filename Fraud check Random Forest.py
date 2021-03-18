# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:02:02 2020

@author: Lenovo
"""

import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv(r"C:\Users\Lenovo\Documents\Assignments\Fraud_check.csv")

# data['Taxable_Income']=data['Taxable_Income'].replace()
data.loc[data['Taxable_Income']<=30000,'Taxable_Income']=0
data.loc[data['Taxable_Income']>30000,'Taxable_Income']=1

data=pd.get_dummies(data,columns=['Undergrad','Marital_Status','Urban'],drop_first=True)
# print(data.head(10))
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
data_norm = norm_func(data.iloc[:,1:])
# print(data_norm.tail(10))
x=data_norm
y=data.Taxable_Income
# print(x.head(10))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
model=RandomForestClassifier(n_jobs=2,n_estimators=5000,criterion='entropy')
model.fit(x_train,y_train)
predi=model.predict(x_test)
print(confusion_matrix(y_test,predi))
print(classification_report(y_test,predi))
print(np.mean(predi==y_test))