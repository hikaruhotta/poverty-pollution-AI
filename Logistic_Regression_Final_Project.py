#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler 


# In[85]:


# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 123
learning_rate = 0.01
num_epochs = 10000
batch_size = 256

# Architecture
num_features = 4
num_classes = 13


# In[86]:


#import dataset
df = pd.read_csv('binaryBuckets.csv')
data = df[['PM2.5 Weighted Mean 24-hr', 'CO 2nd Max 8-hr', 'SO2 2nd Max 24-hr', 'PM10 Mean 24-hr']] #select features

#normalize feature data
scaler = StandardScaler() 
scaler.fit(data)

#add binary data
for i in range(num_classes):
    data[i] = df[str(i)]

#prepare training and test data
msk = np.random.rand(len(df)) < 0.7
train = data[msk]
test = data[~msk]

xtrain = np.array(train.iloc[:,0:num_features], dtype=np.float32)
ytrain = np.array(train.iloc[:, num_features:(num_features + num_classes + 1)], dtype=np.float32)

x_train = torch.from_numpy(xtrain)
y_train = torch.from_numpy(ytrain)

xtest = np.array(test.iloc[:,0:num_features], dtype=np.float32)
ytest = np.array(test.iloc[:, num_features:(num_features + num_classes + 1)], dtype=np.float32)

x_test = torch.from_numpy(xtest)
y_test = torch.from_numpy(ytest)


# In[87]:


#Define our model/neural network
model = torch.nn.Sequential(
    torch.nn.Linear(num_features, num_features, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(num_features, num_features, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(num_features, num_classes, bias=True),
    torch.nn.ReLU(),
    torch.nn.Softmax(dim=1)
)
print(model)


# In[88]:


#define loss function and optimizer
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#gradient descent
for epoch in range(num_epoch):
    input = Variable(x_train)
    target = Variable(y_train)

    # forward
    out = model(input)
    loss = loss_function(out, target)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # show
    print('Epoch[{}/{}], loss: {:.6f}'
          .format(epoch + 1, num_epoch, loss.data.item()))


# In[94]:


#Validation over the training dataset to get an accuracy score
counter = 0
for i in range(len(x_test)):
    result = model(torch.tensor([[x_test[i][0],  x_test[i][1],  x_test[i][2],  x_test[i][3]]], dtype=torch.float32))
    predY = 0
    probY = 0
    for j in range(13):
        prob = float(result[0][j])
        if prob > probY:
            probY = prob
            predY = j
    
    truthY = 0
    for j in range(11):
        if y_test[i][j] == 1:
            truthY = j
            break
            
    if predY == truthY:
        counter+=1
    
print(float(counter)/len(x_test))

