# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:49:02 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
company=pd.read_csv(r"C:\Users\Lenovo\Documents\Assignments\Company_Data.csv")
company.isnull().sum()
company.ShelveLoc.unique
company.Sales.mean()  #Mean is 7.5Approx
plt.boxplot(company["Sales"])
company["sales"]="<-7.5"
company.loc[company["Sales"]<=7.5,"sales"]="Low Sales"
company.loc[company["Sales"]>=7.5,"sales"]="High Sales"
company=company.drop(["Sales"],axis=1)
colnames=list(company.columns)
features=company.iloc[:,0:10]
labels=company.iloc[:,10]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
select_columns=["ShelveLoc","Urban","Urban","US","sales"]
le.fit(company[select_columns].values.flatten())
company[select_columns]=company[select_columns].apply(le.fit_transform)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(features,labels,test_size=0.3,stratify = labels) 

from sklearn.ensemble import RandomForestClassifier as RF
model=RF(n_jobs=4,n_estimators=150,oob_score=True,criterion='entropy')
model=model.fit(x_train,y_train)

train_pred=model.predict(x_train)

train_conf=confusion_matrix(y_train,train_pred)

from sklearn.metrics import accuracy_score
train_accu=accuracy_score(y_train,train_pred)#100


test_pred=model.predict(x_test)


test_con=confusion_matrix(y_test,test_pred)

test_accu=accuracy_score(y_test,test_pred)#81

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
colnames=list(company.columns)
predictors=colnames[:10]
target=colnames[10]
tree_1=model.estimators_[20]
dot_data=StringIO()
export_graphviz(tree_1,out_file=dot_data,rounded=True,feature_names=predictors,class_names=target,impurity=False,proportion=False,precision=2)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
pwd

graph.write_png("companyrf.png")
graph.write_pdf("companyrf.pdf")