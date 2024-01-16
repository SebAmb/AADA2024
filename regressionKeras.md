# Premier projet en Keras - Regression logistique

Nous nous plaçons dans le cadre d'une régression logistique donc dans le cas d'un **classifieur binaire**.

Les étapes que vous apprendrez dans ce TP sont les suivantes :

Charger des données
Définir le modèle Keras
Compiler le modèle Keras
Compatible avec le modèle Keras.
Évaluer le modèle Keras
Attachez le tout ensemble
Faire des prédictions

Vous utiliserez l'environnement Colab Research de Google: https://colab.research.google.com

Vous devez vous y connecter.

La première étape consiste à définir les fonctions et les classes que vous aller utiliser.

Vous utiliserez la bibliothèque NumPy et Sklearn pour charger votre jeu de données et deux classes de la bibliothèque Keras pour définir votre modèle.

Les importations requises sont répertoriées ci-dessous

```
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import seaborn as sns

import pandas as pd
```

## Chargement des données

Commencer par charger la dataset :
```
iris = datasets.load_iris()
print(iris.target)
```
Vous pouvez trouver des informations concernant cette dataset ici : https://scikit-learn.org/stable/datasets/index.html#iris-dataset

**Question : analyser la structure des données. Combien d'échantillons ? Combien de paramètres ?**

Pour produire un modèle et le tester sur les données, il faut transformer l'objet iris en numpy array. Attention, cette base de données comporte 
trois classes : 0, 1, 2. Pour se placer dans le cadre d'une classification binaire, nous allons donc dans un premier temps extraire cette base en fusionnant les classes 0 et 1.

```
X = iris["data"][:, 0:4]
print(X)

# On créé une liste qui contient 1 lorsque la fleur est de type 2 et 0 sinon pour faire une classification
y = iris["target"]
print(y)

# Grâce à cette ligne, la classe 2 est étiquetée 1 et les classes 0 et 1 sont étiquetées 0
y = (iris["target"] == 2)
print(y)
```




# Second projet - application du précédent projet à un autre dataset



Vous allez charger les données contenues dans le fichier dataset_diabets.csv

