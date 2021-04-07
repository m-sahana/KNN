#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


zoo=pd.read_csv("D:\done\KNN\zoo.csv")


# In[3]:


zoo


# In[5]:


import matplotlib.pyplot as plt
plt.hist(zoo['type'])
plt.xlabel("types")
plt.ylabel("distance")


# In[25]:


import seaborn as sns
plt.figure(figsize=(12,6))
sns.scatterplot(x=zoo['animal name'], y = zoo['type'])


# In[26]:


sns.boxplot(zoo['type'])


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


train,test = train_test_split(zoo,test_size = 0.2)


# In[8]:


from sklearn.neighbors import KNeighborsClassifier as KNC


# In[9]:


neigh = KNC(n_neighbors= 7)
neigh.fit(train.iloc[:,1:17],train.iloc[:,17])
train_acc_1 = np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])
train_acc_1


# In[10]:


test_acc_1= np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])
test_acc_1


# In[11]:


neigh = KNC(n_neighbors= 7)
neigh.fit(train.iloc[:,1:17],train.iloc[:,17])
train_acc_2 = np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])
train_acc_2


# In[12]:


test_acc_2 = np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])
test_acc_2


# In[13]:


acc=[]


# In[14]:


for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:17],train.iloc[:,17])
    train_acc = np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])
    test_acc = np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])
    acc.append([train_acc,test_acc])


# In[15]:


import matplotlib.pyplot as plt


# In[16]:


plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")


# In[17]:


plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")


# In[ ]:




