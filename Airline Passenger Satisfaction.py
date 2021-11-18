#!/usr/bin/env python
# coding: utf-8

# In[113]:


from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import pyplot as plt
import scipy.stats

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
train_dataset = load_dataset('train.csv')
train_data = train_dataset.values

test_dataset = load_dataset('test.csv')
test_data = test_dataset.values


# In[114]:


train_dataset


# In[115]:


# Anem a veure quants valors nul tenim al train
print("Per comptar el nombre de valors no existents:")
print(train_dataset.isnull().sum())


# In[116]:


# Veiem que tenim 310 nuls de 103904 a 'Arrival Delay in Minutes'. El que faré és eliminar aquestes dades, ja que són molt poques comparant amb el total 
train_dataset = train_dataset.dropna().copy()


# In[117]:


# Anem a veure quants valors nul tenim al test
print("Per comptar el nombre de valors no existents:")
print(test_dataset.isnull().sum())


# In[118]:


# Veiem que tenim 83 nuls de 25976 a 'Arrival Delay in Minutes'. El que faré és eliminar aquestes dades, ja que són molt poques comparant amb el total  
test_dataset = test_dataset.dropna().copy()


# In[ ]:





# In[119]:


# Anem a veure com es distribueixen les dades
import plotly.express as px
fig= px.sunburst(train_dataset, path=['Customer Type','Type of Travel','satisfaction','Class'], values='Flight Distance',color='Age')
fig.show()


# In[120]:


# Veiem que la majoria de "Disloyal Costumers" és gent jove, tots viatgen per treball, i molt pocs estan satisfets
# Per la gent que viatge per viatges personals, veiem que molts voten "neutral or dissatisfaied", peròi casi tots viatgen amb classe econòmica
# Molts dels "loyal Costumers" que viatgen per treball estan satisfets


# In[121]:


# Mirem de quin tipus són els atributs
print("Per visualitzar informació de la BBDD:")
train_dataset.info()


# In[122]:


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

cnt_gender = test_dataset['Gender'].value_counts().to_frame()
cnt_customer_type = test_dataset["Customer Type"].value_counts().to_frame()
cnt_type_travel = test_dataset['Type of Travel'].value_counts().to_frame()
cnt_class = test_dataset["Class"].value_counts().to_frame()
cnt_satisfaction = test_dataset['satisfaction'].value_counts().to_frame()
from sklearn.preprocessing import  LabelEncoder
le=LabelEncoder()
test_dataset.iloc[:,2] = le.fit_transform(test_dataset.iloc[:,2])
test_dataset.iloc[:,3] = le.fit_transform(test_dataset.iloc[:,3])
test_dataset.iloc[:,5] = le.fit_transform(test_dataset.iloc[:,5])
test_dataset.iloc[:,6] = le.fit_transform(test_dataset.iloc[:,6])
test_dataset.iloc[:,24] = le.fit_transform(test_dataset.iloc[:,24])

test_data = test_dataset.values


# In[123]:


# Eliminar les columnes "Unnamed: 0" i "id" del dataset del train i del test, ja que no ens aporten res
train_dataset.drop("Unnamed: 0",axis=1,inplace=True)
train_dataset.drop("id",axis=1,inplace=True)
test_dataset.drop("Unnamed: 0",axis=1,inplace=True)
test_dataset.drop("id",axis=1,inplace=True)


# In[124]:


train_dataset


# In[125]:


# Defineixo els atributs dependents (X) i els atributs independents (Y)
x_train = train_data[:, :-1]
y_train = train_data[:, -1] 

print("Dimensionalitat de la BBDD de train:", train_dataset.shape) #shape of the data
print("Dimensionalitat de les entrades X de train", x_train.shape)
print("Dimensionalitat de l'atribut Y de train", y_train.shape)
print(" ")

x_test = test_data[:, :-1]
y_test = test_data[:, -1] 

print("Dimensionalitat de la BBDD de test:", test_dataset.shape) #shape of the data
print("Dimensionalitat de les entrades X de test", x_test.shape)
print("Dimensionalitat de l'atribut Y de test", y_test.shape)


# In[126]:


print("Per visualitzar les primeres 5 mostres de la BBDD de train:")
train_dataset.head() 


# In[127]:


print("Per visualitzar les primeres 5 mostres de la BBDD de test:")
test_dataset.head() 


# In[128]:


print("Per visualitzar informació de la BBDD de train:")
train_dataset.info()


# In[129]:


print("Per visualitzar informació de la BBDD de test:")
test_dataset.info()


# In[130]:


print("Per veure estadístiques dels atributs numèrics de la BBDD de train:")
train_dataset.describe()


# In[132]:


# El màxim del "Departure Delay in Minutes" és molt gran (1592 minuts)
# El màxim del "Arrival Delay in Minutes" també és molt gran (1584 minuts)
# Anem a veure si aquest valors són normals, o si són casos puntuals. Ho farem a través de boxplots


# In[141]:


departure = sns.boxplot(x=train_dataset['Departure Delay in Minutes'])
departure


# In[142]:


arrival = sns.boxplot(x=train_dataset['Arrival Delay in Minutes'])
arrival


# In[144]:


# Veiem que, efectivament, aquests valors pertanyen a outliers.
# Com que el nostre dataset és molt gran i no volem que aquests outliers ens esbiaixin el nostre dataset, els eliminem
outliers = train_dataset[train_dataset['Arrival Delay in Minutes'] > 1250].index
train_dataset.drop(outliers, inplace=True)


# In[145]:


print("Per veure estadístiques dels atributs numèrics de la BBDD de test:")
test_dataset.describe()


# In[146]:


# Anem a veure les correlacions
import seaborn as sns
plt.figure(figsize=(20, 15))
sns.heatmap(train_dataset.corr(), cmap='RdYlGn', annot = True)
plt.title("Correlation between Variables")
plt.show()


# In[147]:


# Anem a veure quines variables estan més correlacionades amb la nostra variable y ('satisfaction')
train_dataset.corr().abs()['satisfaction'].sort_values(ascending = False)


# In[ ]:





# In[ ]:




