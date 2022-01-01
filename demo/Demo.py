#!/usr/bin/env python
# coding: utf-8

# In[86]:


import dill
import joblib
import pandas as pd
import numpy as np


# In[87]:


classificador = joblib.load("classificador.pkl")
mitjana = joblib.load("mean.pkl")
desviacio_estandard = joblib.load("desviacio_estandard.pkl")


# A continuació omplirem un seguit de dades (que són les que introdueix el passatger) per predir quina valoració donaria el passatger de l'aerolínia

# In[88]:


noves_dades = []


# In[89]:


# 'Gender' (0 = female, 1 = male)

gender = #Introduir dada
noves_dades.append(gender)


# In[90]:


# 'Customer Type' (0 = Loyal Customer, 1 = Disloyal Customer)

customer_type = #Introduir dada
noves_dades.append(customer_type)


# In[91]:


# 'Age' 

age = #Introduir dada
noves_dades.append(age)


# In[92]:


# 'Type of Travel' (0 = Business Travel, 1 = Personal Travel)

type_travel = #Introduir dada
noves_dades.append(type_travel)


# In[93]:


# 'Class' (0 = Business, 1 = Eco, 2 = Eco Plus)

classe = #Introduir dada
noves_dades.append(classe)


# In[94]:


# 'Flight distance'

flight_distance = #Introduir dada
noves_dades.append(flight_distance)


# In[95]:


# 'Inflight wifi service' (0 = not applicabele, 1-5)

wifi_service = #Introduir dada
noves_dades.append(wifi_service)


# In[96]:


# 'Departure/Arrival time convenient' (0 = not applicabele, 1-5)

time_convenient = #Introduir dada
noves_dades.append(time_convenient)


# In[97]:


# 'Ease of Online booking' (0 = not applicabele, 1-5)

online_booking = #Introduir dada
noves_dades.append(online_booking)


# In[98]:


#'Gate location' (0 = not applicabele, 1-5)

gate_location = #Introduir dada
noves_dades.append(gate_location)


# In[99]:


# 'Food and drink' (0 = not applicabele, 1-5)

food_drink = #Introduir dada
noves_dades.append(food_drink)


# In[100]:


# 'Online boarding' (0 = not applicabele, 1-5)

online_boarding = #Introduir dada
noves_dades.append(online_boarding)


# In[101]:


# 'Seat comfort' (0 = not applicabele, 1-5)

seat = #Introduir dada
noves_dades.append(seat)


# In[102]:


# 'Inflight entertainment' (0 = not applicabele, 1-5)

entertainment = #Introduir dada
noves_dades.append(entertainment)


# In[103]:


# 'On-board service' (0 = not applicabele, 1-5)

onboard_service = #Introduir dada
noves_dades.append(onboard_service)


# In[104]:


# 'Leg room service' (0 = not applicabele, 1-5)

leg_room = #Introduir dada
noves_dades.append(leg_room)


# In[105]:


# 'Baggage handling' (1-5)

baggage = #Introduir dada
noves_dades.append(baggage)


# In[106]:


# 'Check-in service' (0 = not applicabele, 1-5)

checkin = #Introduir dada
noves_dades.append(checkin)


# In[107]:


# 'Inflight service' (0 = not applicabele, 1-5)

inflight_service = #Introduir dada
noves_dades.append(inflight_service)


# In[108]:


# 'Cleanliness' (0 = not applicabele, 1-5)

cleanliness = #Introduir dada
noves_dades.append(cleanliness)


# In[109]:


# 'Departure Delay in Minutes'

departure_delay = #Introduir dada
noves_dades.append(departure_delay)


# In[110]:


# 'Arrival Delay in Minutes'

arrival_delay = #Introduir dada
noves_dades.append(arrival_delay)


# In[112]:


# 'Satisfaction' (0 = neutral or dissatisfied, 1 = satisfied)

satisfaction = #Introduir dada
noves_dades.append(satisfaction)


# Ara dividiré les dades, ja que el que volem és predir l'atribut 'Satisfaction'

# In[113]:


dades = np.array(noves_dades)
x = dades[:-1]
y = dades[-1]


# Normalitzem les dades amb la mitjana i la desviació estàndard de les dades d'entrenament

# In[114]:


scaled_x = (x - mitjana)/desviacio_estandard


# In[115]:


x2 = [scaled_x]
x3 = np.array(x2)


# In[116]:


prediccio = classificador.predict(x3)
print(prediccio)


# In[121]:


y2 = [y]
y3 = np.array(y2)


# In[122]:


joblib.dump(prediccio,'prediccio.pkl')
joblib.dump(y3,'y_real.pkl')

