#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# In[3]:


# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
train_dataset = load_dataset('train.csv')


# In[4]:


# Eliminem els valors null
train_dataset = train_dataset.dropna().copy()


# In[5]:


# Convertim els atributs de tipus 'objecte' en números (del train_dataset i del test_dataset) 
cnt_gender = train_dataset['Gender'].value_counts().to_frame()
cnt_customer_type = train_dataset["Customer Type"].value_counts().to_frame()
cnt_type_travel = train_dataset['Type of Travel'].value_counts().to_frame()
cnt_class = train_dataset["Class"].value_counts().to_frame()
cnt_satisfaction = train_dataset['satisfaction'].value_counts().to_frame()
from sklearn.preprocessing import  LabelEncoder
le=LabelEncoder()
train_dataset.iloc[:,2] = le.fit_transform(train_dataset.iloc[:,2])
train_dataset.iloc[:,3] = le.fit_transform(train_dataset.iloc[:,3])
train_dataset.iloc[:,5] = le.fit_transform(train_dataset.iloc[:,5])
train_dataset.iloc[:,6] = le.fit_transform(train_dataset.iloc[:,6])
train_dataset.iloc[:,24] = le.fit_transform(train_dataset.iloc[:,24])

train_data = train_dataset.values


# In[6]:


# Eliminar les columnes "Unnamed: 0" i "id" del dataset del train i del test, ja que no ens aporten res
train_dataset.drop("Unnamed: 0",axis=1,inplace=True)
train_dataset.drop("id",axis=1,inplace=True)


# In[7]:


# Eliminem outliers
outliers = train_dataset[train_dataset['Arrival Delay in Minutes'] > 1250].index
train_dataset.drop(outliers, inplace=True)


# In[8]:


# Defineixo els atributs dependents (X) i els atributs independents (Y) de la base de dades del train i del test
train_data = train_dataset.values
x_train = train_data[:, :-1]
y_train = train_data[:, -1] 


# In[12]:


# Estandaritzem les dades
sc = StandardScaler()
scaler = sc.fit(x_train)
mean = scaler.mean_
stnd_deviation = scaler.scale_

scaled_xtrain = sc.fit_transform(x_train)


# In[16]:


# Guardem les dades netes i estandaritzades

joblib.dump(scaled_xtrain,'scaled_xtrain.pkl')
joblib.dump(y_train,'y_train.pkl')
joblib.dump(mean,'mean.pkl')
joblib.dump(stnd_deviation,'desviacio_estandard.pkl')

