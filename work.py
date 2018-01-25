import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('50_Startups.csv')
dataset=dataset.drop(['State'],axis=1)
dataset=pd.concat([city,dataset],axis=1)

x=dataset.iloc[:,0:5].values
y=dataset.iloc[:,5:6].values

city=pd.get_dummies(dataset['State'],drop_first=True)

#splitting the data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
ref=LinearRegression()
ref.fit(x_train,y_train)

pred=ref.predict(x_test)




