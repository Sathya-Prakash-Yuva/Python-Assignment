# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 11:58:30 2020

@author: Lenovo
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data=pd.read_csv("../input/glass.csv")
X=data[['K','Al']]
y=data['Type']
print(X.shape)
#from sklearn.cross_validation import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
rangeList=list(range(1,50))
scorelist=[]
for k in rangeList:
    knn=KNeighborsClassifier(n_neighbors=k)
    cap_score=cross_val_score(knn,X,y,cv=5,scoring='accuracy').mean()
    scorelist.append(cap_score)
import matplotlib.pyplot as pl
pl.plot(rangeList,scorelist)
pl.xlabel('KNN Range')
pl.ylabel('Accuracy Score')
#from sklearn.linear_model import LogisticRegression
#lr=LogisticRegression()
#lr.fit(X_train,y_train)
#y_lr_pred=lr.predict(X_test)
#print("Logistic Regression result ",metrics.accuracy_score(y_lr_pred,y_test))