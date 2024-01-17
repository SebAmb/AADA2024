# Premier projet en Keras - Regression logistique - Suppléments

Dans ce sujet, nous allons étudier quelques spécificités de Keras et notamment les points suivants :

1. Spécifier un sous-ensemble de validation
2. Afficher les courbes d'apprentissage
3. Tester différents algorithmes d'apprentissage
4. Enregistrer/charger les poids et l'architrcture du réseau
5. Optimisation des données d'apprentissage
6. Outil Tensorboard

Pour cela vous reprendrez les lignes de code que vous avez développés dans le sujet précédent
pour faire un script complet d'un classifieur MLP binaire dans lequel vous assurerez les étapes suivantes :

1. Chargement de la dataset IRIS avec sa modification sur deux classes
2. Définition et compilation de l'architecture du réseau MLP
3. Lancement de l'entrainement
4. Evaluation du modèle obtenu

## 

Durant l'apprentissage, les données de la base sont présentées au réseau un nombre __n__ de fois (__n__ epochs). La base est découpée en plusieurs
batchs, chaque batch contenant un nombre __m__ d'échantillons. Par conséquent, pour une base d'apprentissage de __L__ échantillons
et si toute la base est utilisée pour apprendre le modèle, à chaque epoch __L/m__ batchs sont présentés au réseau. Un batch présenté correspond à une itération. 

Lorsque vous lancez la méthode ```model.fit(X, y, epochs=..., batch_size=...)``` de votre modèle alors vous obtenez la sortie suivante :

```
Epoch 1/200
15/15 [==============================] - 1s 4ms/step - loss: 0.6153 - accuracy: 0.7133
Epoch 2/200
15/15 [==============================] - 0s 3ms/step - loss: 0.5432 - accuracy: 0.8533
```

**RAPPEL** Dans cette configuration, à chaque itération, sont calculées la valeur de loss et la valeur d'accuracy, toutes les deux calculées à partir de la base d'apprentissage.
Les valeurs de loss sont calculées en utilisant la fonction loss précisée lors de l'appel de la méthode compile() du modèle. Les valeurs d'accuracy sont calculées en utilisant
la métrique précisée lors de l'appel de la méthode compile().

Lors de l'apprentissage il est requis de préciser une base de données de validation. Voici deux manières de le faire.

_Méthodes 1_ : préciser le paramètre __validation_split__ entre 0 et 1. Une valeur de 0.2 définit une base de validation composée de 20% des échantillons de la base d'apprentissage.

```
history=model.fit(X, y, epochs=500, batch_size=10, shuffle=False, validation_split=0.0)
```
La sortie précise alors les valeus de loss et d'accuracy pour la base d'apprentissage restante (i.e. 80%) et la base de validation. Rappelons que seule les 80% servent à
mettre à jour directement ls poids du réseau. La base de validation est utilisée pour qualifier le pouvoir de généralisation du réseau. Cette qualification s'effectue 
à chaque epoch. La sortie de obtenue est la suivante :
```

Epoch 5/500
12/12 [==============================] - 0s 6ms/step - loss: 0.0325 - accuracy: 0.9917 - val_loss: 0.3693 - val_accuracy: 0.7667
Epoch 6/500
12/12 [==============================] - 0s 5ms/step - loss: 0.0327 - accuracy: 0.9917 - val_loss: 0.3862 - val_accuracy: 0.7667
```



