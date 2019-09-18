#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 


# In[2]:


data = pd.read_csv('Project/mergedData.csv')
data = data.set_index('County Code')
scaler = StandardScaler() 
data = data[['PM2.5 Weighted Mean 24-hr', 'CO 2nd Max 8-hr', 'SO2 2nd Max 24-hr', 'PM10 Mean 24-hr']]
data


# In[3]:


kmeans_data = data.values
kmeans = KMeans(n_clusters=5).fit(kmeans_data)
y_kmeans = kmeans.predict(kmeans_data)
kmeans_data


# In[4]:


plt.scatter(kmeans_data[:, 0], kmeans_data[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.xlabel("CO 2nd Max 8-hr")
plt.ylabel("SO2 2nd Max 24-hr")
plt.title("K-Means CO and SO2 Clusters")
plt.show()

