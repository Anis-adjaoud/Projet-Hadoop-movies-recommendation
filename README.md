# Système de Recommandation de Films

Notre projet vise à concevoir un système intelligent de recommandation de films, combinant analyse de données massives, machine learning, et visualisation interactive, tout ça sur graçe à un cluster Hadoop configuré.
Il s'appuie sur un fichier (movie_rank.csv) issu d’IMDb contenant les films, leurs notes, genres, durées et votes. Exemple de ligne :

--------------------------------------------------------------------------------------------------------------------------------------------
Column1 tconst    titleType primaryTitle originalTitle isAdult startYear runtimeMinutes genres                    averageRating    numVotes
--------------------------------------------------------------------------------------------------------------------------------------------
141081	tt0145487 movie	    Spider-Man   Spider-Man    0	   2002	     121        	Action,Adventure,Sci-Fi	  7.4              911741.0
--------------------------------------------------------------------------------------------------------------------------------------------

L’objectif est double :

-   Explorer les tendances par genre, note, époque ou popularité
-   Recommander des films pertinents grâce à des modèles de clustering et de similarité.

---

## Motivations

Le choix de ce thème s’explique par plusieurs raisons :

-   La saturation de contenu rend la recherche de films de qualité difficile.
-   Un système de recommandation intelligent répond à ce besoin en personnalisant les choix.
-   C’est un sujet grand public, parlant à tout le monde, facile à illustrer et à évaluer.

---

## Installations

On a commencé par monter un cluster Hadoop sur docker, avec un namenode, deux datanodes pour la réplication des données, et un jupyter avec spark installé.
les fichiers docker-compose.yaml et config sont utilisés pour cette installation.
Ensuite, on écrit le fichier de notre dataset sur HDFS, afin de faire des traitements Spark par la suite : 
    ==> docker cp .\Dataset\movie_rank.csv projet_hadoop-namenode-1:/tmp/movie_rank.csv
    ==> hdfs dfs -mkdir -p /user/data/
    ==> hdfs dfs -put /tmp/movie_rank.csv /user/data/
En faisant hdfs dfs -ls -R /, on voit que notre fichier est bien présent sur HDFS.

Maintenant que notre architecture est préte, on accède au serveur jupyter sur : http://127.0.0.1:8888/lab, qui nous permet de créer des fichiers python, jupyter, ouvrir un terminal...

---

## Composants du projet

### `movies_recommendation.ipy` – Recommandation intelligente avec Spark

#### Étapes clés :

-   **Nettoyage et filtrage des données** :

    -   Suppression des films adultes et des valeurs non valides
    -   Cast des types, filtrage sur les années, votes, genres, durée

-   **Création de variables avancées** :

    -   `bayesian_rating` : note pondérée selon le volume de votes
    -   `popularity_score` : normalisation logarithmique du nombre de votes
    -   `quality_score` : combinaison pondérée des deux précédentes
    -   Catégorisation par :
        durée : - `Short` si durée du film < 90mn, - `Medium` si durée du film < 120mn, - `Long` si durée du film < 150mn
        époque : - `Classic` si année < 1960 - `Vintage` si année < 1980 - `Modern` si année < 2000 - `Contemporary` si année < 2010 - sinon `Recent`

-   **Clustering KMeans** :

    -   Normalisation des features
    -   Groupement des films par profils pour affiner les recommandations

-   **Système KNN hybride (numérique + genre)** :

    -   Pondération : 30% caractéristiques numériques, 70% genres
    -   Utilisation d’un encodage one-hot et de la distance cosine
    -   Score final combiné avec genre, qualité et similarité

-   **Interface interactive** :
    -   Recommandation par genre
    -   Recommandation à partir d’un film donné
    -   Recherche personnalisée (multi-filtres)
    -   Top films par époque ou catégorie

---

### `dataViz.py` – Visualisation avec Streamlit

#### Fonctionnalités :

-   Top 10 des films par genre et année
-   Classement pondéré des genres selon note × log(votes)
-   Top 20 des films les plus votés
-   Filtres interactifs : année, genre
-   Graphiques dynamiques avec Plotly

#### Étapes techniques :

-   Nettoyage Spark (`\\N`, cast, explode des genres)
-   Agrégation des votes et notes par genre
-   Classement avec `Window.partitionBy()` + `row_number()` pour obtenir les tops par genre
-   Conversion vers Pandas pour l'affichage avec Plotly

---

## Calculs utilisés

### **1. Bayesian rating** :

    bayesian = ((v*r) + (m*R)) / (v+m)

-   _v_ : nombre de votes du film
-   _r_ : note du film
-   _m_ : seuil (ex. 90ᵉ percentile des votes)
-   _R_ : note moyenne globale

---

### **2. Popularity (log-normalized)** :

    (log(v+1) - log(min+1)) / (log(max+1)-log(min+1))

-   Donne un score de popularité normalisé entre 0 et 1

---

### **3. Score pondéré par genre** :

    score pondéré = note moyenne * log10(votes)

-   Utilisé pour classer les genres en fonction de leur qualité perçue et leur popularité

---

### **4. Score composite KNN** :

    score final = 0.6 * similarité globale + 0.3 * overlap genre + 0.1 * (quality score / 10)

-   Combine distance cosine, ressemblance de genre et qualité intrinsèque du film

---

## Exécution du projet

# Lancer le système de recommandation intelligent (jupyter) :
movies_recommendation.ipy (Cell 1)

# Lancer l’interface de visualisation Streamlit :
streamlit run dataViz.py

```