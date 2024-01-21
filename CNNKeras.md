# Mon premier réseau de neurones convolutifs en Keras

Dans les deux sujets précédents vous avez appris à définir, entraîner, évaluer et inférer un réseau de neurones totalement connectés que vous
avez appliqué pour assurer une tâche de regression logistique (sur le jeu de données Iris) et de classification multiclasse (sur 
le jeu de données Iris et MNIST). Vous avez également connaissance de quelques commandes pour afficher les courbes d'apprentissage afin dé déterminer
laqualité de l'entraînement et l'apparition de surapprentissage.

Dans ce sujet, nous nous plaçons dans le cadre d'une application de classifications d'images et nous allons apprendre à définir l'architecture d'un réseau de neurones convolutifs (CNN) et à l'entraîner.
Nous commencerons par faire ce travail sur le jeu de données MNIST. Puis nous l'étendrons à d'autres jeu de données.

## Chargement et mise en place des données

**Question : commencer par importer les librairies donc vous aurez besoin puis charger les images MNIST comme indiqué dans le sujet précédent. Le jeu d'entraînement avec ses labels
seront chargés dans les variables x_train, y_train. Le jeu de test avec ses labels seront chargés dans les variables x_test, y_test.**

Puis nous allons afficher 25 images du jeu de données d'entrainement.

```
# MNIST
class_names = ['ZERO', 'UN', 'DEUX', 'TROIS', 'QUATRE', 'CINQ',
               'SIX', 'SEPT', 'HUIT', 'NEUF']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i]])
plt.show()
```

**Question : faites de même avec l'ensemble des images de tests**

## Spécification du modèle convolutif

Un réseau de neurones convolutifs de base est structuré en deux parties. La premmière partie du réseau réalise la projection des données qui lui sont placées en entrée, dans un espace de "caractéristiques" (feature map) : elle fournit ainsi un vecteur d'informations qui "résume" l'information contenue dans les données d'entrée au regard d'une tâche à réaliser (une classification d'images par exemple). La second partie du réseau utilise un réseau de neurones complètement connecté (MLP) pour réaliser la tâche proprement dite. Dans ce premier exemple, nous allons définir un CNN qui sera composé d'une seule couche de convolution et d'une couche de __pooling__. La couche de convolution sera composé de 32 filtres (32 noyaux ou kernels) de taille 3x3; elle fera appel à la fonction ```layers.Conv2D()``` et à une fonction d'activation de type Relu. La couche de __pooling__ fera appel à l'opérateur ```MaxPooling2D()``` sur un voisinage 2x2. Voici la défintion de l'architecture neuronale de manière similaire à ce que nous avons fait précédemment :

```
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Couche "feature map"
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.summary()

# Couche classification
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

```
La couche de convolution 2D prend en entrée une image de taille 28x28 en niveau de gris (input_shape=(28,28,1)).
Sachant que par défaut le paramètre __stride = (1,1)__ et le paramaètre __padding='valid'__ sont passés à la fonction Conv2D, la sortie de cette couche est de taille 26x26x32, 32 correspondant au nombre de filtres.
__'valid'__ signifie qu'aucun padding n'est appliqué, donc les effets de bord de la convolution subsustent et entrainent une baisse de la résolution.

La couche de pooling (2,2) réduit la résolution par 2 dans chaque dimension en affectant la valeur maximale sur un voisinage 2x2. Là encore par défaut le paramètre de __padding='valide'__ ce qui fait que la sortie de cette couche est de résolution (13x13x32).

Dans la second partie du réseau, nous retrouvons la couche __Flatten__ qui permet d'applatir la sortie de la couche précédente.

**Question : quelle est la taille de la sortie de la couche __Flatten__?**

**Question : quel est le nombre de paramètres du modèle complet ?**

La phase d'entraînement va permettre de déterminer la valeur de ces paramètres.

## Entraînement du modèle

L'objectif est de déterminer les poids du modèle.

**Question : sur la base de ce que vous avez fait dans le TP précédent, écrire les lignes de code pour définir les paramètres de l'apprentissag et pour lancer le fitting du modèle.**






