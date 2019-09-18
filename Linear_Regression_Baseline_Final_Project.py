#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[12]:


#Import data
data = pd.read_csv('bucketed.csv')
df = data[['PM2.5 Weighted Mean 24-hr', 'CO 2nd Max 8-hr', 'SO2 2nd Max 24-hr', 'PM10 Mean 24-hr']]

#Normalize data
scaler = StandardScaler() 
scaler.fit(data)
df['Income_Bracket'] = data['Income_Bracket']
data = df
X = data.drop('Income_Bracket', axis = 1)

#Find coefficients - not very important
lm = LinearRegression()
lm.fit(X, data.Income_Bracket)
coeff = pd.DataFrame(zip(X.columns, lm.coef_), columns = {'features', 'estimatedCoefficients'})

#Split training and test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, data.Income_Bracket, test_size = 0.3, random_state = 5)

#Set up Linear Regression model and Train it
lm = LinearRegression()
lm.fit(X_train, Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)

#Compute MSE
Y_train = Y_train.values.astype(float)
Y_test = Y_test.values.astype(float)
print('Fit a model X_train, and calculate MSE with X_test, Y_test: ', np.mean((Y_test - lm.predict(X_test)) ** 2))

#Compute Accuracy
result = lm.predict(X_test)
result = [int(round(i)) for i in result]
counter = 0
for i in range(len(Y_test)):
    if float(Y_test[i]) == result[i]:
        counter+=1

print('Accuracy: ',float(counter)/len(Y_test))
   


# In[ ]:




