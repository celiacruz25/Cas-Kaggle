#!/usr/bin/env python
# coding: utf-8

# In[6]:


import dill
import joblib
from sklearn.metrics import accuracy_score


# In[7]:


prediccio = joblib.load("prediccio.pkl")
y_real = joblib.load("y_real.pkl")


# In[8]:


accuracy = accuracy_score(y_real, prediccio)


# In[9]:


print(accuracy)

