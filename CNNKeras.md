# My first CNN in Keras

Dans les deux sujets précédents vous avez appris à définir, entraîner, évaluer et inférer un réseau de neurones totalement connectés que vous
avez appliqué pour assurer une tâche de regression logistique (sur le jeu de données Iris) et de classification multiclasse (sur 
le jeu de données Iris et MNIST). Vous avez également connaissance de quelques commandes pour afficher les courbes d'apprentissage afin dé déterminer
laqualité de l'entraînement et l'apparition de surapprentissage.

Dans ce sujet, nous nous plaçons dans le cadre d'une application de classifications d'images et nous allons apprendre à définir un réseau de neurones convolutifs.
Nous commencerons par étudier à nouveau le jeu de données MNIST.

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

Comme vous le savez, un réseau de neurones convolutifs de base est structuré en deux parties. La premmière partie opère la projection des données placées en entrée dans un espace de "caractéristiques" (feature map) et fournit un vecteur d'informations 
et la seconde partie

