#!/usr/bin/env python
# coding: utf-8

# In[4]:


import dill
import joblib
from xgboost import XGBClassifier


# In[5]:


scaled_xtrain = joblib.load("scaled_xtrain.pkl")
y_train = joblib.load("y_train.pkl")


# In[6]:


xgboost = XGBClassifier()
xgboost.fit(scaled_xtrain, y_train)


# In[7]:


joblib.dump(xgboost,'classificador.pkl')

