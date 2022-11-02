# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 17:26:47 2020

@author: prash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing dataset
test_data=pd.read_csv('test.csv')
train_data=pd.read_csv('train.csv')
x=train_data.iloc[:,:-2].values
y=train_data.iloc[:,9:11].values
#fittig null/missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(strategy="mean")
imputer=imputer.fit(x[:,3:4])
x[:,3:4]=imputer.transform(x[:,3:4])
#taking care of categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
for i in [0,1,2,4]:
    x[:,i]=labelencoder_x.fit_transform(x[:,i])

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(x, y)

z=test_data.iloc[:,:].values
z[:,3:4]=imputer.transform(z[:,3:4])
for i in [0,1,2,4]:
    z[:,i]=labelencoder_x.fit_transform(z[:,i])

y_pred=regressor.predict(z)
zz=test_data.iloc[:,0].values
f=open("sol.csv",'w')
f.write("pet_id,breed_category,pet_category\n")
for i in range(len(zz)):
    f.write(str(zz[i])+","+str(int(round(y_pred[i][0])))+","+str(int(round(y_pred[i][1])))+"\n")
f.close()

