#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler 


# In[38]:


# Architecture
num_features = 4
num_classes = 13
learning_rate = 0.01
num_epochs = 5000

#get input and target tensors
data = pd.read_csv('binaryBuckets.csv')
df = data[['PM2.5 Weighted Mean 24-hr', 'CO 2nd Max 8-hr', 'SO2 2nd Max 24-hr', 'PM10 Mean 24-hr', '0', '1','2','3','4','5','6','7','8','9','10','11', '12']]
scaler = StandardScaler() 
scaler.fit(df)

#Split training and test data
msk = np.random.rand(len(df)) < 0.7
trainingData = df[msk]
testData = df[~msk]


dataTrain = torch.tensor(trainingData.values, dtype=torch.float)
x_train, y_train = dataTrain[:,0:num_features], dataTrain[:, num_features:(num_features + num_classes + 1)]

dataTest = torch.tensor(testData.values, dtype=torch.float)
x_test, y_test = dataTest[:,0:num_features], dataTest[:, num_features:(num_features + num_classes + 1)]


# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, batch_size = num_features, num_features, num_classes, len(y_train)


# In[39]:


#Define Multi-Layered Linear Regression
model = nn.Sequential(nn.Linear(n_in, n_h),
                     nn.ReLU(),
                     nn.Linear(n_h, n_h),
                      nn.ReLU(),
                    nn.Linear(n_h, n_out), 
                      nn.Sigmoid())


# In[40]:


# Construct the loss function
criterion = torch.nn.BCELoss()

# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[41]:


# Gradient Descent
for epoch in range(num_epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)

    # Compute and print loss
    loss = criterion(y_pred, y_train)
    print('epoch: ', epoch,' loss: ', loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer.step()


# In[42]:


correct = 0
counter = 0
for row in x_test:
    result = model(row)
    highest = torch.argmax(result)
    index = torch.argmax(y_test[counter])
    if highest == index:
        correct+=1
    counter+=1
    
print(float(correct)/len(y_test))

