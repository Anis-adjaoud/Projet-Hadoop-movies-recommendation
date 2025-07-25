from pyspark.sql import SparkSession
import streamlit as st
import plotly.express as px
from pyspark.sql.functions import col, split, when, explode, row_number, floor,count,avg, log10 
from pyspark.sql.window import Window

# Initialisation de Spark
spark = SparkSession.builder \
    .appName("App HDFS") \
    .getOrCreate()

# Chargement du fichier movie_rank.csv
df = spark.read.csv(
    "hdfs://namenode:8020/user/data/movie_rank.csv",
    header=True,
    inferSchema=True,
    sep=","
)

# Nettoyage de runtimeMinutes
df_clean = df.withColumn(
    "runtimeMinutes", 
    when(col("runtimeMinutes") != "\\N", col("runtimeMinutes").cast("int"))
)

# Nettoyage de genres
df_clean = df_clean.withColumn(
    "genres", 
    when(col("genres") != "\\N", split(col("genres"), ","))
)
 
df_clean = df_clean\
    .withColumn("startYear", when(col("startYear") != "\\N", col("startYear").cast("int")))\ # Nettoyage de startYear
    .withColumn("averageRating", col("averageRating").cast("float")) # Conversion de averageRating en float

# Conversion de averageRating en float
df_clean = df_clean.withColumn(
    "averageRating", 
    col("averageRating").cast("float")
)

# Conversion de numVotes en int
df_clean = df_clean.withColumn(
    "numVotes", 
    col("numVotes").cast("int")
)

# Suppression des lignes contenant des valeurs nulles dans certaines colonnes
df_clean = df_clean.na.drop(subset=["primaryTitle", "genres", "startYear", "averageRating"])

# Ajout d'un titre à l'application
st.title("📽️ Analyse de Films")

# Création d'un DataFrame avec un genre par ligne
df_exploded = df_clean.withColumn("genre", explode(col("genres"))) \
                .filter(col("genre") != "Adult") \
                .filter(col("startYear") > 1919 )

# Récupération de la liste des genres uniques
genre_list = (
    df_exploded.select("genre")
    .distinct()
    .orderBy("genre")
    .rdd.map(lambda row: row["genre"])
    .collect()
)

# Filtre par année
year_list = (
    df_exploded.select("startYear")
    .distinct()
    .orderBy("startYear")
    .rdd.map(lambda row: row["startYear"])
    .collect()
)
selected_year = st.selectbox("Sélectionnez une année", year_list)

# Ajout d'un filtre pour sélectionner un genre
selected_genre = st.selectbox("Sélectionnez un genre", genre_list)

# Filtrage des données par le genre et l'année sélectionnés
filtered_data = df_exploded.filter(
    (col("genre") == selected_genre) & (col("startYear") == selected_year)
)
# Création d'un top 10 des films pour ce genre
top_movies = filtered_data.orderBy(col("averageRating").desc()).limit(10)

# Conversion en pandas pour Plotly
top_movies_pd = top_movies.select("primaryTitle", "averageRating", "startYear").toPandas()

# Affichage du graphique
st.subheader(f"🏅 Top 10 des films du genre {selected_genre}")
fig = px.bar(
    top_movies_pd, 
    x="primaryTitle", 
    y="averageRating",
    text="averageRating",
    hover_data=["startYear"],
    color="averageRating",
    color_continuous_scale="RdYlGn",
    title=f"Top 10 des films du genre {selected_genre} par note moyenne"
)
fig.update_layout(
    xaxis_title="Titre du film",
    yaxis_title="Note moyenne",
    xaxis={'categoryorder': 'total descending'}
)
fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

###########################################################################

# Classement pondéré des genres (note × popularité)
df_genre_score = df_clean.withColumn("genre", explode(col("genres")))

# AJOUT DU FILTRE pour exclure "adult" et "\N"
df_genre_score = df_genre_score.filter(
    (col("genre") != "Adult") & (col("genre") != "\\N")
)

# Calcul de la note moyenne et du nombre de votes moyen par genre
df_genre_score = df_genre_score.groupBy("genre").agg(
    avg("averageRating").alias("note_moyenne"),
    avg("numVotes").alias("votes_moyens")
).filter(col("votes_moyens") > 10)  

# Score pondéré : note * log10(votes)
df_genre_score = df_genre_score.withColumn(
    "score_pondere", col("note_moyenne") * log10(col("votes_moyens"))
)

df_genre_score_sorted = df_genre_score.select(
    "genre", "note_moyenne", "votes_moyens", "score_pondere"
).orderBy(col("score_pondere").desc())

# Conversion uniquement pour l'affichage
df_score_pd = df_genre_score_sorted.toPandas()

# Visualisation du classement pondéré
st.subheader("🏆 Classement des genres par score pondéré (note × popularité)")
fig_score = px.bar(
    df_score_pd.sort_values("score_pondere", ascending=False),
    x="genre",
    y="score_pondere",
    hover_data=["note_moyenne", "votes_moyens"],
    color="score_pondere",
    title="Genres les plus appréciés et populaires (score pondéré)",
    labels={
        "genre": "Genre",
        "score_pondere": "Score pondéré",
        "note_moyenne": "Note moyenne",
        "votes_moyens": "Votes moyens"
    },
    color_continuous_scale="Blues"
)
fig_score.update_layout(xaxis_title="Genre", yaxis_title="Score pondéré")
st.plotly_chart(fig_score, use_container_width=True)

###########################################################################

df_votes = df_exploded.filter(col("numVotes").isNotNull())

windowSpec_votes = Window.partitionBy("genre").orderBy(col("numVotes").desc())

df_ranked_votes = df_votes.withColumn("rank", row_number().over(windowSpec_votes))

df_top10_votes_per_genre = df_ranked_votes.filter(col("rank") <= 10)

df_top_votes_all = df_clean.filter( col("numVotes").isNotNull())

df_top20_votes =df_top_votes_all.orderBy( col("numVotes").desc()).limit(20)

# conversion en pandas pour affichage
df_top20_votes_pd = df_top20_votes.select("primaryTitle", "numVotes", "averageRating").toPandas()

# bar chart des 20 films les plus votés 
st.subheader("🏆 Top 20 des films les plus votés (tous genres confondus)")

fig_top20_votes = px.bar(
    df_top20_votes_pd,
    x="primaryTitle",
    y="numVotes"  ,
    text="numVotes",
    hover_data=["averageRating"],
    title="Top 20 des films les plus populaires par nombre de votes",
    labels={"primaryTitle": "Titre du film", "numVotes": "Nombre de votes"},
    color="numVotes",
    color_continuous_scale="Tealgrn"
)

fig_top20_votes.update_layout  ( xaxis_tickangle=45, yaxis_tickformat=".",yaxis_title="Nombre de votes", xaxis_title="Titre")
fig_top20_votes.update_traces(texttemplate='%{text:.0f}', textposition='outside')  # valeurs au dessus des barres sans décimales 
st.plotly_chart(fig_top20_votes, use_container_width=True)