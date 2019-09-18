#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib2
import requests

print('Beginning file download with requests')

states = ["al", "ak", "az", "ar", "ca", "co", "ct", "dc", "de", "fl", "ga", 
          "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", 
          "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", 
          "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", 
          "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"]

for state in states:
    for year in range (1997, 2018):
        url = 'https://www2.census.gov/programs-surveys/saipe/datasets/'
        url += str(year) + '/' + str(year) + '-state-and-county/est' + str(year)[-2:] + '-' + state + '.dat'  
        r = requests.get(url)
        savePath = 'Project/income_data/' + str(year) + '-' + state + '.csv'
        with open(savePath, 'wb') as f:  
            f.write(r.content)
print("Done downloading")


# In[1]:


columnIndex = ['FIPS State Code', 'FIPS County Code', 'All Ages Poverty', 
               '90% Lower Bound All Ages', 
               '90% Upper Bound All Ages',
               'Percent All Ages',
               '90% Lower Bound Percent',
               '90% Upper Bound Percent',
               '0-17', 
               '90% Lower Bound 0-17', 
               '90% Upper Bound 0-17', 
               'Percent 0-17',
               '90% Lower Bound Percent 0-17',
               '90% Upper Bound Percent 0-17', 
               '5-17',
               '90% Lower Bound 5-17',
               '90% Upper Bound 5-17',
               'Percent 5-17',
               '90% Lower Bound Percent 5-17',
               '90% Upper Bound Percent 5-17',
               'MedianHouseholdIncome', 
               '90% Lower Bound Median Household Income',
               '90% Upper Bound Median Household Income', 'County', 'Class', 'State']

states = ["al", "ak", "az", "ar", "ca", "co", "ct", "dc", "de", "fl", "ga", 
          "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", 
          "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", 
          "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", 
          "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"]


# In[2]:


import pandas as pd
incomeDictionary = dict()

for year in range (2010, 2011):
    dataFrames = dict()
    for state in states:
        path = 'Project/income_data/' + str(year) + '-' + state + '.csv'
        data = pd.read_csv(path, delim_whitespace=True, header=0)
        data = data.drop(list(data)[-4:], axis=1)
        data = data.drop(list(data)[-1:], axis=1)
        if len(data.columns) > 26:
            data = data.drop(list(data)[26 - len(data.columns):], axis=1)
        data
        data.columns = columnIndex
        data.insert(0, "County Code", data['FIPS State Code']*1000 + data['FIPS County Code'] , True) 
        data = data.drop(list(data)[1], axis=1)
        data = data.drop(list(data)[1], axis=1)
        #data = data.drop(columns=['All Ages Poverty', '90% Lower Bound All Ages', '90% Upper Bound All Ages',
                         #'0-17', '90% Lower Bound 0-17', '90% Upper Bound 0-17', 
                          #'5-17','90% Lower Bound 5-17','90% Upper Bound 5-17'])
        data = data[['County Code','MedianHouseholdIncome']]
        dataFrames[state] = data
    incomeDictionary[year] = dataFrames


# In[6]:


perYear = dict()
for year in range (2001, 2002):
    yearData = pd.DataFrame(columns=['County Code', 'MedianHouseholdIncome'])
    for state in states:
        yearData = yearData.append(incomeDictionary[year][state])
    perYear[year] = yearData  


# In[17]:


import numpy as np
AQIDictionary = {}
start_year = 2000
YEARS = [start_year+i for i in range(18)]
for year in YEARS:
    df = pd.read_csv('Project/AQI_data/conreport' + str(year) + '.csv')
    df = df.drop(list(df)[2], axis=1)
    df = df.drop(list(df)[3], axis=1)
    df = df.drop(list(df)[4], axis=1)
    df = df.drop(list(df)[5], axis=1)
    df = df.drop(list(df)[6], axis=1)
    df = df.drop(list(df)[6], axis=1)
    df = df.drop(list(df)[7], axis=1)
    df = df.drop(list(df)[-1:], axis=1)
    df = df.replace('.', np.nan)
    df = df.dropna(thresh=6)
    AQIDictionary[year] = df


# In[18]:


import pandas as pd
pop_density = pd.read_csv('Project/pop_density_data/pop_density_2010.csv')
pop_density = pop_density.set_index('County Code')
pop_density


# In[19]:


AQIDictionary[2000] = AQIDictionary[2000].set_index('County Code')
AQIDictionary[2000]


# In[20]:


two_thou = pd.read_csv('Project/2000.csv')
two_thou = two_thou.drop(list(two_thou)[0], axis=1)
two_thou = two_thou.set_index('County Code')
two_thou


# In[21]:


merged = AQIDictionary[2000].merge(two_thou, left_index=True, right_index=True)
merged = merged.merge(pop_density, left_index=True, right_index=True)
merged


# In[383]:


almost = pd.DataFrame()
for year in YEARS:
    almost = almost.append(concatnated[year])


# In[22]:


data = merged.drop('County', axis=1)
#values = {'CO 2nd Max 8-hr':3.2, 'NO2 Mean 1-hr':0, 'Ozone 4th Max 8-hr':0, 'SO2 2nd Max 24-hr':0, 'PM2.5 Weighted Mean 24-hr':0, 'PM10 Mean 24-hr':0}
values = {'CO 2nd Max 8-hr':3.2, 'NO2 Mean 1-hr':17, 'Ozone 4th Max 8-hr':0.081, 'SO2 2nd Max 24-hr':22, 'PM2.5 Weighted Mean 24-hr':13.85, 'PM10 Mean 24-hr':26}
data = data.fillna(value=values)
data


# In[23]:


import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[24]:


X = data.drop('MedianHouseholdIncome', axis = 1)
lm = LinearRegression()
lm.fit(X, data.MedianHouseholdIncome)
coeff = pd.DataFrame(zip(X.columns, lm.coef_), columns = {'features', 'estimatedCoefficients'})
coeff


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(
    X, data.MedianHouseholdIncome, test_size = 0.33, random_state = 5)
print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape


# In[26]:


lm = LinearRegression()
lm.fit(X_train, Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)


Y_train = Y_train.values.astype(float)
Y_test = Y_test.values.astype(float)

print('Fit a model X_train, and calculate MSE with Y_train: ', np.mean((Y_train - lm.predict(X_train)) ** 2))

print('Fit a model X_train, and calculate MSE with X_test, Y_test: ', np.mean((Y_test - lm.predict(X_test)) ** 2))


# In[ ]:


Baselines
- random guessing
- majority classfier
- logistic regression - sklearn
- 1 layer neural network 

Oracle
- cheating
- literature (paper with same data)
- derive weakly linked relationship

Proposal
- why is this proble interesting
- Have you thought about it more
- How 


# In[ ]:





# In[208]:


import sklearn
import numpy as np
import seaborn as sns



# In[ ]:




