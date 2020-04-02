import numpy as np
import sklearn
import scipy as sp
import pandas as pd
import matplotlib
import seaborn as sns
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
desired_width = 320
pd.set_option('display.width', desired_width)
from sklearn.datasets import load_breast_cancer
titanic_df=pd.read_csv('titanic/train.csv')
titanic_df.head(10)
titanic_df.drop(['PassengerId','Name','Ticket','Cabin'],'columns',inplace=True)
titanic_df.dropna(inplace=True)
fig,ax = plt.subplots(figsize=(12,8))
plt.scatter(titanic_df['Fare'], titanic_df['Survived'])
plt.xlabel('Age')
plt.ylabel('Survived')
fig,ax = plt.subplots(figsize=(12,10))
titanic_data_corr =titanic_df.corr()
sns.heatmap(titanic_data_corr,annot=True)

label_encoding=preprocessing.LabelEncoder()
titanic_df['Sex']=label_encoding.fit_transform(titanic_df['Sex'])