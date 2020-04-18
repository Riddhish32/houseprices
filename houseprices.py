# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

os.chdir(r'C:\Users\riddh\Downloads')

data_frame= pd.read_csv(r"C:\Users\riddh\Downloads\train.csv")

print(data_frame.isna())

data_frame= data_frame[['LotArea','OverallQual','OverallCond','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','SalePrice']]

data_frame.corr()

data_frame.info()


sns.heatmap(data_frame.corr())

data_test= pd.read_csv(r"C:\Users\riddh\Downloads\test.csv")

data_test= data_test.dropna(axis=0, inplace=True)

data_test=data_test[['LotArea','OverallQual','OverallCond','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea']]

y=data_frame[['SalePrice']]

X=data_frame[['LotArea','OverallQual','OverallCond','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea']]

data_test.info()

regressor= LinearRegression()

regressor.fit(X,y)

y_pred= regressor.predict(data_test)

y_predtest= regressor.predict(X)

y_predtrain= regressor.predict(data_test)

