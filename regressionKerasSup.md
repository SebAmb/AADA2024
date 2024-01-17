# Premier projet en Keras - Regression logistique - Suppléments

Dans ce sujet, nous allons étudier quelques spécificités de Keras et notamment les points suivants :

1. Afficher les courbes d'apprentissage
2. Tester différents algorithmes d'apprentissage
3. Spécifier un sous-ensemble de validation
4. Enregistrer/charger les poids et l'architrcture du réseau
5. Optimisation des données d'apprentissage
6. Outil Tensorboard

Pour cela vous reprendrez les lignes de code que vous avez développés dans le sujet précédent
pour faire un script complet d'un classifieur MLP binaire dans lequel vous assurerez les étapes suivantes :

1. Chargement de la dataset IRIS avec sa modification sur deux classes
2. Définition et compilation de l'architecture du réseau MLP
3. Lancement de l'entrainement
4. Evaluation du modèle obtenu

## Affichage des courbes d'apprentissage

Durant l'apprentissage, les données de la base sont présentées au réseau un nombre __n__ d'épochs sous la forme
de batch, chaque batch contenant un nombre __m__ d'échantillons. Par conséquent, pour une base d'apprentissage de __L__ échantillons
et si toute la base est utilisée pour apprendre le modèle, à chaque epoch __L/m__ batchs sont présentés au réseau.

Lorsque vous lancez la méthode fit() de votre modèle alors vous obtenez la sortie suivante :

```
Epoch 1/200
15/15 [==============================] - 1s 4ms/step - loss: 0.6153 - accuracy: 0.7133
Epoch 2/200
15/15 [==============================] - 0s 3ms/step - loss: 0.5432 - accuracy: 0.8533
```




