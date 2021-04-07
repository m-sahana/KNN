#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


glass=pd.read_csv("D:\done\KNN\glass.csv")
glass


# In[4]:


import matplotlib.pyplot as plt


# In[8]:


plt.hist(glass['Type'])
plt.xlabel("types")
plt.ylabel("distance")


# In[20]:


import seaborn as sns
sns.boxplot(glass['Type'])


# In[9]:


from sklearn.model_selection import train_test_split
train,test = train_test_split(glass,test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier as KNC


# In[10]:


neigh=KNC(n_neighbors=7)
neigh.fit(train.iloc[:,1:9],train.iloc[:,9])
train_acc_1 = np.mean(neigh.predict(train.iloc[:,1:9])==train.iloc[:,9])
train_acc_1


# In[11]:


test_acc_1= np.mean(neigh.predict(test.iloc[:,1:9])==test.iloc[:,9])
test_acc_1


# In[12]:


neigh=KNC(n_neighbors=7)
neigh.fit(train.iloc[:,1:9],train.iloc[:,9])
train_acc = np.mean(neigh.predict(train.iloc[:,1:9])==train.iloc[:,9])
train_acc


# In[13]:


test_acc = np.mean(neigh.predict(test.iloc[:,1:9])==test.iloc[:,9])
test_acc


# In[14]:


acc=[]


# In[15]:


for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:9],train.iloc[:,9])
    train_acc = np.mean(neigh.predict(train.iloc[:,1:9])==train.iloc[:,9])
    test_acc = np.mean(neigh.predict(test.iloc[:,1:9])==test.iloc[:,9])
    acc.append([train_acc,test_acc])


# In[16]:


import matplotlib.pyplot as plt


# In[17]:


plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")


# In[18]:


plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")


# In[ ]:




