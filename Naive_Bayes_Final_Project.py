#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler 
import pandas as pd

# Architecture
num_features = 4
num_classes = 1

df = pd.read_csv('bucketed.csv')
data = df[['PM2.5 Weighted Mean 24-hr', 'CO 2nd Max 8-hr', 'SO2 2nd Max 24-hr', 'PM10 Mean 24-hr', 'Income_Bracket']] #select features
scaler = StandardScaler() 
scaler.fit(data)

#prepare training and test data
msk = np.random.rand(len(df)) < 0.7
train = data[msk]
test = data[~msk]

xtrain = np.array(train.iloc[:,0:num_features], dtype=np.float32)
ytrain = np.array(train.iloc[:, num_features:(num_features + num_classes + 1)], dtype=np.float32)
xtest = np.array(test.iloc[:,0:num_features], dtype=np.float32)
ytest = np.array(test.iloc[:, num_features:(num_features + num_classes + 1)], dtype=np.float32)

#Model
clf = MultinomialNB()
clf.fit(xtrain, ytrain)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)


# In[16]:


#Validation
prediction = clf.predict(xtest)
correct = 0
for i in range(len(prediction)):
    if prediction[i] == ytest[i]:
        correct+=1
print(float(correct)/len(prediction))

