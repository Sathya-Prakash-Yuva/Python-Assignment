# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:05:10 2020

@author: Lenovo
"""

import pandas as pd 
import numpy as np 
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv(r"C:\Users\Lenovo\Documents\Assignments\Company_Data.csv")
print(data.info())
# data['d']=pd.cut(data['Sales'], range(0,17,2))
input()
print(data.corr())
data.loc[data['Sales']<10,'Sales']=0
data.loc[data['Sales']>=10,'Sales']=1

data=pd.get_dummies(data,columns=['ShelveLoc','Urban','US'],drop_first=True)
print(data.head())
input()
print(data.corr())
x=data.drop('Sales',axis=1)
y=data.Sales
print(x,y)
input()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
model=RandomForestClassifier(n_jobs=2,n_estimators=200,criterion='entropy')
model.fit(x_train,y_train)
predi=model.predict(x_test)
print(confusion_matrix(y_test,predi))
print(classification_report(y_test,predi))
print(np.mean(predi==y_test))
# 94 acuracy