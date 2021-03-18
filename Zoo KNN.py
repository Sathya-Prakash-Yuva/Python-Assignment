# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 11:41:58 2020

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
##############################################################################

# Import CSV (Change filepath in speech marks to where CSV is saved)
Zoo = pd.read_csv(r"C:\Users\Lenovo\Documents\Assignments\ZooDataset.csv")

# Convert CSV to dataframe
Data = pd.DataFrame(Zoo)

# Feature matrix
X = np.array(Data.loc[:,["Hair","Feathers", "Eggs", "Milk", "Airborne", 
             "Aquatic","Predator", "Toothed", "Backbone", "Breathes", 
             "Venomous","Fins", "Legs", "Tail", "Domestic", "Catsize"]])

# Target we want to predict
y = np.array(Data.loc[:,"Type"])

# Array for predictions
prdList = [[]]

# Array for y[test]
yList = [[]]

# Creates array to store means
MeanList = []

# Mean array is currently empty
MeanTotal = 0

# Runs KFold 100 times
for i in range(100):
    
    # Creates array to store scores
    ScoreList = []

    # Score array is currently empty
    Total = 0

    # KFold instance, does 10 folds
    kf = KFold(n_splits=10, random_state=None, shuffle=True)

    # KNN instance using 3 neighbours (default) for each feature
    knn = KNeighborsClassifier()

    # Split data into training and testing within K-Fold
    for train, test in kf.split(X):
    
        # Fit test and train arrays
        fit = knn.fit(X[train], y[train])
    
        # Create prediction
        prd = knn.predict(X[test])
        
        # Append predictions to array
        prdList.append(prd)
        
        # Append y[test] to array
        yList.append(y[test])
    
        # Add score to array
        ScoreList.append(accuracy_score(y[test],prd))
    
        # Display accuracy score
        print("The accuracy score is", accuracy_score(y[test],prd))

    # Add each score as value to 'List' array
    for val in ScoreList:
        Total += val
    
    # Creates a list of 10 means
    MeanList.append(Total/len(ScoreList))
    
    # Total of all the means
    MeanTotal += (Total/len(ScoreList))
    
    # Display mean accuracy score     
    print ("\nThe mean accuracy score is", Total/len(ScoreList))

    # Display confusion matrix
    print ("\nConfusion matrix:\n", 
           confusion_matrix(train[y], knn.predict(train[X])), "\n")
    
    # Sort the list of means in ascending order
    MeanList.sort()
    
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Display list of all 100 means
print("\nList of means:")
for i in range(len(MeanList)):
    print(MeanList[i])
    
# Display mean of all 100 means    
print("\nThe grand mean is", MeanTotal/len(MeanList))

# Show lists of predicted and actual animal types
print("\nPredicted animal types:",prdList[1]) 
print("Actual animal types:",yList[1])

# Display graph showing predicted vs actual types
plt.title("Predicted vs actual animal types for one K-Fold sample\n")
plt.plot(prdList[1], 'b', label = 'prediction')
plt.plot(yList[1], 'r', label = 'y_test')
plt.xlabel("Sample")
plt.ylabel("Animal type")
plt.show()