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
- A l'atribut 'Gender' podem veure que el tant per cent d'homes satisfets és molt semblant al de dones satisfetes, i que tant en el cas dels homes com en el de les dones hi ha més valoracions del tipus 'neutre o no satisfactori'. Per tanat, podem pensar que que aquest atribut no ens donarà molta informació sobre si el client triarà una valoració de la satisfacció bona o dolenta.
- A l'atribut 'Customer Type' podem veure que tant en el cas de clients lleials o no lleials, el nombre de valoracions neutrals o no saisfactories és molt alt. Tot i així, la diferència de valoracions satisfactories contra les neutrals o no satisfactories és major en els clients no lleials que en els lleials.
- A l'atribut 'Type of Travel' veiem que tenim molts més clients que viatgen per negocis que per gust. Podem observar que dels clients que viatgen per negocis, hi ha més de satisfets (casi el 60%!) que de neutrals o no satisfets. En canvi, dels que viatgen per que volen hi ha més de neutrals o no satisfets (casi 90%) que satisfets.
- A l'atribut 'Class' podem veure, la única classe en la que la majoria de passatgers estàn satisfets és la business. En la que estan menys satisfets és en la classe econònica.
He volgut entretenir-me en veure a cada classe quin 'Type of Travel' hi va, ja que hem vist que la gent que viatja per negocis acostuma a posar una valoració positiva de l'aerolínia, i justament la classe 'Business' és l'única on hi ha més gent amb valoracions positives que negatives. El que he pogut observar és que efectivament a la classe 'Business' és on hi va més gent que viatja per negocis i on hi ha menys gent que treballa per assumptes personals, i a la resta de classes és al contrari. Per tant, podriem dir que la causa de que hi hagi més passatger que viatgen per negocis amb valoracions positives que negatives és perquè viatgen en classe 'Business', mentre que entre la de gent que viatge per temes personales hi ha més amb valoracions negatives que positives perquè viatgen en classe 'Eco' o 'Eco Plus'. Així doncs, hauriem d'intentar millorar l'estada dels passatgers de les classes 'Eco' i 'Eco Plus'.
- A la resta d'atributs categòrics ('Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness') he pogut observar que els únics casos on hi havia més gent que posava una bona valoració de l'aerolínia eren quan els passatgers puntuaven aquell servei concret amb les màximes puntuacion (4 o 5) o bé quan posaven 0 (que vol dir que no han contestat aquella pregunta). Això és lògic, ja que una persona que hagi tingut una bona experiència en un servei és més probable que acabi valorant positivament l'aerolínia que no pas una persona que hagi tingut una mala experiència. Pel cas en que hi ha més gent que no ha valorat el servei que ha posat una bona valoració de l'aerolínia podríem pensar que el que passa és que per aquest passatger aquell servei no té importància (o com per exemple el cas del servei del wifi, que no l'hagi fet servir), i han estat altres factors que han fet que valori positivament l'aerolínia.
- A l'atribut'satisfaction' podem veure que més ho menys hi ha el mateix nombre de passatgers satisfets que passatgers neutrals o insatisfets. Per tant, podem dir que el nostre atribut objectiu està balancejat.

Pel que fa a les estadístiques dels atributs quantitatius, hem observat el següent:
- A l'atribut 'Age' la gent jove i la gent d'avançada edat acostumen a estar més satisfets dels vols. Són la gent d'entre més o menys 40 i 60 anys que donen més valoracions neutrals o de no satisfacció.
- A l'atribut 'Flight distance' a mesura  que  la distància del vol augmenta, la satisfacció decreix.
- Els atributs 'Departure Delay in Minutes' i 'Arrival Delay in Minutes' estàn molt correlacionats, i graficant-los veiem que tene una relació linial.

### Model
Hem provat diferents models, i més d'un ha donat resultats molt bons. 

|MODEL|HYPERPARÀMETRES|ACCURACY|TEMPS|
|--|--|--|--|
|Logistic Regression|Default|0.8710|0.3053|
|Decision Tree|max_depth=13, random_state=42|0.9549|0.5662|
|Random Forest|max_depth=25, random_state=0, n_estimators= 1200|0.9632|141.9058|
|XGBoost|Default|0.9629|3.68607|
|KNN|Default|0.9289|0.0120|
|SVM|random_state=2|0.9552|224.3337|

Desprès de comparar el temps, els accuracy i la ROC CUrve i la Precision-Recall Curve de tots els models, sembla que el que dona millors resultats és el XGBoost.

### PCA
Un alrte experiment que he fet és provar de fer un PCA amb el model XGBoost per veure fins quan podríem reduir les dimensions de l'espai.
En el resultat podem veure que l'accuracy del model va decreixent a poc a poc a mida que reduïm la dimensió. Però podem observar un canvi bastant dràstic a partir de quan la dimensió és de 5 cap avall.

### Atributs més important
Amb el model XGBoost hem fet un estudi dels atributs que tenen més quan un passatger pren la  decisió de donar una valoració positiva o negativa de l'aerolínia, i hem vist que aquests són ,per ordre d'importància:
1. 'Arrival Delay in Minutes'
2. 'Departure Delay in Minutes'
3. 'Cleanliness'
4. 'Inlfight service'
5. 'Check-in service'
6. 'Baggage handling'
7. 'Leg room service'
8. 'On-board sevice'
9. 'Inflight entertainment'
10. 'Seat comfort'
11. 'Online boarding'
12. 'Food and drink'
13. 'Gate location'
14. 'Esea of Online booking'
15. 'Departure/Arrival time convenient'
16. 'Inflight wifi service'
17. 'Flight distance'
18. 'Class'
19. 'Type of Travel'
20. 'Age'
21. 'Custom Type'
22. 'Gender'

## Demo
Per tal de fer una prova, es pot executar el fitxer "Demo.py".

## Conclusions
Hem vist que el model que millor prediu el nostre atribut objectiu és el XGBoost.
A més , hem observat que si volem ue els passatgers de l'aerolínia la valorin bé, hauríem d'intentar que la sortida i l'arribada dels vols fos puntual, i millorar la neteja i el servei a bord en els vols.
