**Plan détaillé du projet Computer Vision**

Ce plan est conçu pour être suivi séquentiellement. Chaque étape s'appuie sur les précédentes.

**I. Préparation et exploration initiale**

-   [x] **Compréhension approfondie du sujet :**
    -   [x] Relire attentivement le document PDF pour bien saisir les objectifs, les contraintes et les données fournies.
    -   [x] Clarifier les points ambigus (par exemple, "valeur connue de la gravité" dans la section 4).

-   [x] **Exploration des données :**
    -   [x] Visualiser les vidéos fournies pour comprendre le mouvement des objets et la configuration de la scène.
    -   [x] Examiner les images du damier (checkboard) pour évaluer leur qualité et leur utilité pour l'étalonnage.
    -   [x] Noter les caractéristiques importantes (par exemple, la taille du damier, la luminosité, les angles de prise de vue).

- [x] **Création du projet et import des librairies**
    - [x] Créer un nouveau projet et importer les librairies nécessaire
    - [x] Créer des scripts Python pour chaque grande phase du projet (étalonnage, homographie, suivi, etc.).

**II. Étalonnage de la caméra**

-   [x] **Préparation des images du damier :**
    -   [x] Si nécessaire, prétraiter les images (par exemple, ajuster le contraste, réduire le bruit).
    -   [x] Identifier les coins du damier dans chaque image (manuellement ou à l'aide d'une fonction de détection).

-   [x] **Calibration de la caméra :**
    -   [x] Utiliser les points d'angle détectés et les dimensions réelles du damier pour calibrer la caméra.
    -   [x] Obtenir les paramètres intrinsèques de la caméra (matrice de la caméra, coefficients de distorsion).

-   [x] **Évaluation de la distorsion :**
    -   [x] Visualiser l'effet de la distorsion sur les images du damier.
    -   [x] Appliquer la correction de distorsion et évaluer si l'amélioration est significative.
    -   [x] Décider s'il est nécessaire de compenser la distorsion pour la suite du projet.

**III. Homographie**

-   [ ] **Identification de points de correspondance :**
    -   [ ] Choisir une image de référence à partir de l'une des vidéos.
    -   [ ] Identifier manuellement au moins quatre points de correspondance (non colinéaires) entre l'image de référence et le plan réel de la scène (le tableau blanc).

-   [ ] **Calcul de l'homographie :**
    -   [ ] Utiliser les points de correspondance pour calculer la matrice d'homographie.

-   [ ] **Vérification de l'homographie :**
    -   [ ] Appliquer l'homographie à l'image de référence et vérifier si le plan de la scène est correctement aligné.
    -   [ ] Appliquer l'homographie à d'autres images de la vidéo et vérifier la cohérence.

**IV. Suivi de l'objet**

-   [ ] **Choix d'une méthode de détection :**
    -   [ ] Sélectionner une méthode de détection d'objet adaptée (par exemple, soustraction d'arrière-plan, détection de couleur, template matching).
    -   [ ] Considérer la robustesse, la vitesse et la facilité d'implémentation de chaque méthode.

-   [ ] **Implémentation de la détection :**
    -   [ ] Appliquer la méthode choisie pour détecter l'objet dans chaque image de la vidéo.
    -   [ ] Obtenir la position (centroïde) et la bounding box de l'objet à chaque instant.

-   [ ] **Filtrage et lissage (facultatif) :**
    -   [ ] Si la détection est bruitée, appliquer un filtre (par exemple, filtre de Kalman, moyenne mobile) pour lisser la trajectoire.

**V. Modélisation de la trajectoire**

-   [ ] **Extraction des données de trajectoire :**
    -   [ ] Transformer les coordonnées de l'objet (centroïde) du repère de l'image au repère du monde réel en utilisant l'homographie.
    -   [ ] Obtenir une séquence de points (x, y) représentant la trajectoire de l'objet dans le plan du tableau blanc.

-   [ ] **Ajustement du modèle balistique :**
    -   [ ] Utiliser les données de trajectoire pour ajuster un modèle balistique simple (parabole).
    -   [ ] Estimer les paramètres du modèle (vitesse initiale, angle initial, gravité si inconnue).

-   [ ] **Prédiction de la position future :**
    -   [ ] Utiliser le modèle ajusté pour prédire la position de l'objet à des instants futurs.
    -   [ ] Visualiser la trajectoire prédite et la comparer à la trajectoire réelle.

**VI. Augmentation de la vidéo**

-   [ ] **Incrustation des informations :**
    -   [ ] Ajouter des éléments visuels à la vidéo pour illustrer les résultats :
        -   [ ] Dessiner le centroïde et la bounding box de l'objet.
        -   [ ] Afficher la trajectoire parabolique approximée.
        -   [ ] Indiquer la position future prédite.
- [ ] **Création de la vidéo augmentée**
    - [ ] Combiner ces éléments à la vidéo de base.

**VII. Rapport et présentation**

-   [ ] **Rédaction du rapport technique :**
    -   [ ] Décrire en détail les étapes suivies, les méthodes utilisées, les résultats obtenus et les difficultés rencontrées.
    -   [ ] Inclure des figures, des tableaux et des captures d'écran pour illustrer le travail.
    -   [ ] Fournir une analyse critique des résultats et discuter des améliorations possibles.
-    [ ] **Préparation de la soutenance**
    - [ ] Présenter de façon clair le dérouler du projet et les résultats
-   [ ] **Archivage du projet :**
    -   [ ] Organiser le code source, les données et le rapport de manière claire et concise.

**Conseils supplémentaires :**

*   **Commencez petit :** Ne cherchez pas à tout faire en même temps. Implémentez d'abord les étapes de base, puis ajoutez progressivement des fonctionnalités plus avancées.
*   **Testez régulièrement :** Vérifiez le fonctionnement de votre code à chaque étape. Cela vous permettra d'identifier et de corriger les erreurs plus facilement.
*   **Documentez votre code :** Ajoutez des commentaires pour expliquer ce que fait chaque partie de votre code. Cela vous aidera à vous y retrouver plus tard et facilitera la collaboration si vous travaillez en équipe.
*   **Soyez persévérant :** La Computer Vision peut être complexe. Ne vous découragez pas si vous rencontrez des difficultés. Faites des recherches, posez des questions et essayez différentes approches.

J'espère que ce plan détaillé vous sera utile. N'hésitez pas à me poser d'autres questions si vous avez besoin de précisions sur certaines étapes. Bon courage pour votre projet !
