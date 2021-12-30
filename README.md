# Pràctica Kaggle APC UAB 2021-22
### Nom: Cèlia Cruz Esclera
### DATASET: Airline Passenger Satisfaction
### URL: https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction

## Resum
El dataset que s'utilitza en aquesta pràctica es tracta d'una enquesta de satisfacció pels passatgers d'una aerolínia.
En el Kaggle se'ns faciliten dues bases de dades, una pper fer el train i l'altra per fer el test. Totes dues tenen els mateixos atributs, però tenen dades diferents.
La base de dades 'train.csv' consta de 103904 dades amb 25 atributs, mentre que la base de dades 'test.csv' té 25976 dades 25 atributs.
Els atributs que apareixen als dos datasets són:
  - 'Gender': gènere del passatger (Female, Male).
  - 'Customer Type': tipus de client (Loyal customer, disloyal customer).
  - 'Age': edat actual del passatger.
  - 'Type of Travel': propòsit del viatge del passatger (Personal Travel, Business Travel).
  - 'Class': classe de l'avió en la que viatja el passatger (Busincess, Eco, Eco Plus).
  - 'Flight distance': distància del viatge en avió.
  - 'Inflight wifi service': nivell de satisfacció del servei de wifi de l'avió (0: no s'ha aplicat, 1-5).
  - 'Departure/Arrival time convenient': nivell de satisfacció en l'hora de sortida/arrivada de l'avió (0: no s'ha aplicat, 1-5).
  - 'Ease of Online booking': nivell de satisfacció sobre la facilitat de reservar el vol online (0: no s'ha aplicat, 1-5).
  - 'Gate location': nivell de satisfacció de la localització de la porta d'embarcament (0: no s'ha aplicat, 1-5).
  - 'Food and drink': nivell de satisfacció del menjar i la beguda (0: no s'ha aplicat, 1-5).
  - 'Online boarding': nivell de satisfacció de l'embarcament online (0: no s'ha aplicat, 1-5).
  - 'Seat comfort': nivell de satisfacció de la comoditat del seient (0: no s'ha aplicat, 1-5).
  - 'Inflight entertainment': nivell de satisfacció de l'entreteniment durant el vol (0: no s'ha aplicat, 1-5).
  - 'On-board service': nivell de satisfacció del servei durant el vol (0: no s'ha aplicat, 1-5).
  - 'Leg room service': nivell de satisfacció de l'espai que hi ha als seients per posar els peus (0: no s'ha aplicat, 1-5).
  - 'Baggage handling': nivell de satisfacció de la manipulació de l'equipatge (1-5).
  - 'Check-in service': nivell de satisfacció del servei check-in (0: no s'ha aplicat, 1-5).
  - 'Inflight service': nivell de satisfacció del servei durant el vol (0: no s'ha aplicat, 1-5).
  - 'Cleanliness': nivell de satisfacció de a neteja (0: no s'ha aplicat, 1-5).
  - 'Departure Delay in Minutes': minuts de retard de la sortida del vol.
  - 'Arrival Delay in Minutes': minuts de retard de l'arrivada del vol.
  - 'Satisfaction': nivell de satisfacció de l'aerolínia (satisfied, neutral or dissatisfied).

Com podem veure, la majoria dels atributs són categòrics, ja que tenim 19 atributs categòrics i només 4 numèrics.

### Objectius del dataset
Aquesta base de dades té com a objectiu veure quins són els factors que fan que un passatger estigui satisfet de l'aerolínia.
Per tant, volem entrenar un model que ens pugui predir amb bons resultats si el nivell de satisfacció del passatger sobre l'aerolía serà 'satisfied' o 'neutral or dissatisfied', i així poder estudiar quins atributs tenen més pes quan el passatger valora la seva resposta. 

## Experiments
Durant aquesta pràctica hem realitzat diferents experiments.

### Preprocessament
Abans de començar a aplicar models, hem fet alguns canvis a la nostra base de dades.

Prier de tot, hem mirat quants valors null's tenim tant al nostre dataset del train com al del test. Veiem que al dataset del train tenim 310 nulls's, tots a l'atribut 'Arrival Delayin Minutes'. Com el nombre de dades de passatgers que tenen null's (310) és molt més petit al nombre total de dades que tenim (103904), he decidit eliminar les dades  que continguin null's, EL mateix passa amb el dataset del test, ja que només tenim 83 null's a l'atribut 'Arrival Delayin Minutes', així que he decidit borrar les dades que tinguin null's pel mateix motiu (tenim 83 dades amb null's d'un total de 25976).

A continuació el que he fet ha estat mirar de quin tipus són els atributs, i he obtingut el següent:
- 'Unnamed: 0': int64  
- 'id': int64  
- 'Gender': object 
- 'Customer Type': object 
- 'Type of Travel': object 
- 'Class': object 
- 'Flight Distance': int64  
- 'Inflight wifi service': int64  
- 'Departure/Arrival time convenient': int64  
- 'Ease of Online booking': int64  
- 'Gate location': int64  
- 'Food and drink': int64  
- 'Online boarding': int64  
- 'Seat comfort': int64  
- 'Inflight entertainment': int64  
- 'On-board service': int64  
- 'Leg room service': int64  
- 'Baggage handling': int64  
- 'Checkin service': int64  
- 'Inflight service': int64  
- 'Cleanliness': int64  
- 'Departure Delay in Minutes': int64  
- 'Arrival Delay in Minutes': float64
- 'satisfaction': object

Com podem veure, tenim molts atributs de tipus objecte, cosa que n'interessa, així que el que he fet és passar-los a úmeros. Per exemple, en el cas de l'atribut 'Gender', la resposta 'female' passarà a ser un 0, i la resposta 'male' passarà a ser un 1, i així amb la resta d'atributs de tipus objecte.

Després d'assegurar-nos que cap valor de cap atribut és null i que tots són números, eliminem  les columnes "Unnamed: 0" i "id" del dataset del train i del test, ja que no ens aporten cap informació útil.

Llavors mirem les estadístiques de la base de dades del train, i ens adonem que el nombre màxim de 'Departure Delay in Minutes' és molt gran (1592.000) i que el nombre màxim de 'Arrival Delay in Minutes' és molt gran (1584.000). Comprovem que això és a causa de outliers, així que el que fem és, com que el nostre dataset és molt gran i no volem que aquests outliers ens esbiaixin el nostre dataset, els eliminem. També treiem els outliers del dataset del test.

### Anàlisi de les dades
COm és lògic en el nostre dataset, decidim que la nostra varaible a predir és l'atribut 'satisfaction', ja que volem veure, a partir de tots els factors que afecten al vol, si els clients surten satisfets o no.

Llavors el que comencem a fer és analitzar cada atribut un a un per veure com es relacionen amb l'atribut objectiu 'satisfaction'.  Ho farem de diferent manera pels atributs categòrics i els quantitatius.

Primer analitzarem els atributs categòrics, on mirem les estadístiques del nombre de clients satisfets i no satisfets de cada categoria del atribut.
