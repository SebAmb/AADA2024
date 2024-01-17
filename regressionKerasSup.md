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

## Spécification de la base de validation

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
à chaque epoch grâce aux 4 valeurs loss, accuracy, val_loss et val_accuracy. La sortie obtenue est la suivante :
```

Epoch 5/500
12/12 [==============================] - 0s 6ms/step - loss: 0.0325 - accuracy: 0.9917 - val_loss: 0.3693 - val_accuracy: 0.7667
Epoch 6/500
12/12 [==============================] - 0s 5ms/step - loss: 0.0327 - accuracy: 0.9917 - val_loss: 0.3862 - val_accuracy: 0.7667
```

_Méthode 2_ : il suffit de créer un sous-ensemble de données de validation à partir de la base d'apprentissage et de le passer à la méthode fit() du modèle.
Pour cela nous allons utiliser la librairie sklearn et la fonction train_test_split() au travers des lignes de code suivantes :
```
# Création des sous ensemble de validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)
```
Dans cet exemple, nous créons les deux sous-ensembles X_train et X_val et leurs labels respectifs y_train et y_val. Les échantillons de la base de validation X_val représentent 20% de la base d'apprentissage initiale X. L'extraction est réalisée de manière aléatoire en précisant le seed du générateur de nombres pseudo-aléatoires (random_state=2).En changeant cette valeur, vous obtiendrez des sous-ensembles différents.

## Affichage des courbes d'apprentissage

Voici quelques lignes pour tracer la valeur de loss, accuracy, val_loss et val_accuracy obtenues à chaque époch.
Pour cela, nous devons récupérer ce qui est retourné par la méthode fit().
```
...
history=mpodel.fit(...)
...
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.plot(history.history['val_loss'], label = 'val loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')
```

**Question : Afficher les courbes pour l'architectures MLP à deux couches cachées dont vous modifierez le nombre de neurones selon les configirations suivantes : (64,32) (32,16)(16,8)(8,4).
A partir de ces 4 courbes, il est alors possible de constater ou non l'apparition ou non d'un surapprenissage (overfitting). Un surapprentissage est le résultat d'une inadéquation entre le nombre de poids 
du réseau et le nombre d'instance de la base d'apprentissage. Une fois vos 4 courbes disponibles, appelez-moi pour une explication.**

## Enregistrement/chargement des poids du réseau appris

Une fois la phase d'entraînement terminée, il est possible d'enregistrer les poids du réseau et son architecture grâce à ```model.save("regressionML.keras")```. Cette fonction sauvegarde l'architecture du réseau et les poids du réseau obtenus à la toute dernière itération dans un fichier compressé. Il est possible de ne sauvegarder que les poids ```model.save_weights("regressionMLP.weights.h5")``` dans un fichier au format .h5.

Une fois les sauvegardes effectuées, il est alors possible de charger à nouveau les réseaux pour réaliser d'autres inférences, sur d'autres plateformes et d'autres données. Le fichier .keras permet, en le chargeant, de décrire à la fois l'architecture et les poids. le fichier .h5  permet de ne charger que les poids et nécessite donc de décrire préalablement et manuellement l'architecture du réseau comme vouz l'avez fait jusque là. 

```
# Charger un modèle
restored_model = keras.models.load_model("regressionMLP.keras")

ou

# Charger les poids d'un modèle
# Description du réseau
model = Sequential()
model.add(Dense(128, input_shape=(4,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.load_weights("regressionMLP.weights.h5")
```
Ainsi, il est possible d'inférer le réseau chargé sur de nouvelles données de la manière suivante :
```
# Prédiction sur le sous-ensemble d'apprentissage
res=model.predict(X_train)
res2=restored_model.predict(X_train)

# Prédiction sur deux nouvelles instances X_1 et X_2
X_1=[5.8,2.6,4.0,1.2]
X_2=[6.3,3.3,6.0,2.5]
res1=model.predict([X_1])
res2=model.predict([X_2])
print('classe {0} and classe {1}'.format(round(res1[0,0]),round(res2[0,0])))
```

## Sauvegarde des poids durant l'apprentissage

Il est possible de sauvegarder les poids du réseau durant l'apprentissage. Pour cela, nous devons créer une fonction callback qui sera appelée par la fonction .fit()
Il est possible de sauvegarder les poids estimés à la fin de chaque epoch (https://keras.io/api/callbacks/model_checkpoint/)
Toutefois, pour éconoimser de l'espace mémoire (tout particulièrement pour les gros réseaux) il est possible de sauvegarder les poids du réseau sous conditions. Par exemple, nous
choisirons de sauvegarder uniquement si la valeur de val_loss est plus faible que la valeur obtenue à l'epoch précédente.

```
# construction de la fonction callback qui sauvegarde les poids
# seulement si le modèle est meilleur sur la base de val_loss
checkpoint = keras.callbacks.ModelCheckpoint(filepath="./weights/weights-{epoch:03d}-{val_loss:.4f}.hdf5", monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]

# appel de la nouvelle fonction de fit()
history=model.fit(X, y, epochs=200, batch_size=10, shuffle=False, validation_split=0.2, callbacks=callbacks, verbose=2)
```

Il est possible de changer la valeur qui est monitorée pour créer la condition de sauvegarde : ```fit(...,monitor=val_accuracy,mode="max",verbose=1)```

**Question : vérifier la bonne sauvegarde des poids dans le dossier que vous avez préciseé (ici ./weights/weights-032-0.4321.hdf5)**

**Question : une fois l'entraînement terminé, charger le dernier fichier de poids sauvegardé avec la callback et inférer ce réseau sur la base X_train afin d'en calculer la précision
comme vous l'avez fait dans le sujet précédent**






