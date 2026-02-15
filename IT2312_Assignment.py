# Databricks notebook source

# MAGIC %md
# MAGIC # IT2312 2025 S2 Individual Assignment
# MAGIC ## Big Data Processing – Movie Ratings
# MAGIC
# MAGIC This notebook processes the MovieLens ml-25m dataset using PySpark DataFrames.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1 – Data Ingestion

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Ingest the 3 files: movies.csv, tags.csv, and ratings.csv

# COMMAND ----------

# Read movies.csv
movies_df = spark.read.csv("/FileStore/tables/movies.csv", header=True, inferSchema=True)

# Read tags.csv
tags_df = spark.read.csv("/FileStore/tables/tags.csv", header=True, inferSchema=True)

# Read ratings.csv
ratings_df = spark.read.csv("/FileStore/tables/ratings.csv", header=True, inferSchema=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Print the number of records and the number of columns for each data file

# COMMAND ----------

print(f"movies.csv  - Number of records: {movies_df.count()}, Number of columns: {len(movies_df.columns)}")
print(f"tags.csv    - Number of records: {tags_df.count()}, Number of columns: {len(tags_df.columns)}")
print(f"ratings.csv - Number of records: {ratings_df.count()}, Number of columns: {len(ratings_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2 – Data Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q1. Unique tags and their occurrence counts (excluding specified tags), sorted descending

# COMMAND ----------

from pyspark.sql.functions import col, count, lower

# List of tags to exclude
exclude_tags = ['sci-fi', 'action', 'comedy', 'mystery', 'war', 'politics', 'religion', 'thriller']

# Filter out the excluded tags (case-insensitive), group by tag, count occurrences, sort descending
q1_df = (tags_df
         .filter(~lower(col("tag")).isin([t.lower() for t in exclude_tags]))
         .groupBy("tag")
         .agg(count("*").alias("cnt"))
         .orderBy(col("cnt").desc()))

display(q1_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q2. Movies tagged 'boring' or 'overrated' with title and average rating, sorted ascending by avg rating (top 10)

# COMMAND ----------

from pyspark.sql.functions import avg

# Filter tags for 'boring' or 'overrated'
boring_overrated_tags = tags_df.filter(lower(col("tag")).isin(["boring", "overrated"]))

# Get average rating per movie
avg_ratings = ratings_df.groupBy("movieId").agg(avg("rating").alias("avgRating"))

# Join tags with movies to get titles
q2_df = (boring_overrated_tags
         .select("movieId")
         .distinct()
         .join(movies_df, "movieId")
         .join(avg_ratings, "movieId")
         .select("movieId", "title", "avgRating")
         .orderBy(col("avgRating").asc()))

display(q2_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q3. Movies tagged 'great acting' or 'inspirational' with title and average rating, sorted descending by avg rating (top 10)

# COMMAND ----------

# Filter tags for 'great acting' or 'inspirational'
great_inspirational_tags = tags_df.filter(lower(col("tag")).isin(["great acting", "inspirational"]))

# Join tags with movies to get titles
q3_df = (great_inspirational_tags
         .select("movieId")
         .distinct()
         .join(movies_df, "movieId")
         .join(avg_ratings, "movieId")
         .select("movieId", "title", "avgRating")
         .orderBy(col("avgRating").desc()))

display(q3_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q4. Aggregate movie ratings into ranges with userId, movieId, rating, and tag columns

# COMMAND ----------

from pyspark.sql.functions import when

# Join ratings with tags
ratings_tags_df = ratings_df.join(tags_df, on=["userId", "movieId"], how="inner")

# Create rating_range column
q4_df = (ratings_tags_df
         .withColumn("rating_range",
                     when(col("rating") < 1, "Below 1")
                     .when((col("rating") >= 1) & (col("rating") < 2), "1 to 2")
                     .when((col("rating") >= 2) & (col("rating") < 3), "2 to 3")
                     .when((col("rating") >= 3) & (col("rating") < 4), "3 to 4")
                     .when((col("rating") >= 4) & (col("rating") < 5), "4 to 5")
                     .otherwise("5 and more"))
         .select("userId", "movieId", "rating", "tag", "rating_range"))

display(q4_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q5. Aggregated rating ranges with tags and tag counts, filtered >200, sorted by range asc and count desc

# COMMAND ----------

# Aggregate by rating_range and tag, count tags
q5_df = (q4_df
         .groupBy("rating_range", "tag")
         .agg(count("*").alias("numTag"))
         .filter(col("numTag") > 200)
         .orderBy(col("rating_range").asc(), col("numTag").desc()))

display(q5_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q6. Conclusions
# MAGIC
# MAGIC Based on the data exploration performed above, the following conclusions and actionable recommendations are drawn to address the two business objectives: **identifying new movie genres to target and produce**, and **identifying examples and elements of bad movies to avoid replicating**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 1. Opportunity in Uncommon Genres — Target Niche Audiences with High-Engagement Content
# MAGIC
# MAGIC From **Q1**, after excluding the mainstream genre-related tags (sci-fi, action, comedy, mystery, war, politics, religion, thriller), the most frequently applied user tags are descriptive qualities such as *atmospheric*, *surreal*, *based on a book*, *twist ending*, *funny*, *visually appealing*, *dystopia*, and *dark comedy*. These tags represent the **attributes audiences actively seek out and remember**, rather than broad genre labels. This indicates a strong viewer appetite for movies that deliver distinctive storytelling elements — particularly atmospheric world-building, narrative surprises (twist endings), and visual craftsmanship.
# MAGIC
# MAGIC **Recommendation:** The production company should prioritise films that blend uncommon genre elements with these high-engagement attributes. For example, producing a *visually appealing dystopian thriller with a twist ending* would tap into multiple high-frequency tags simultaneously, increasing discoverability and audience resonance. Rather than competing in saturated mainstream genres, investing in niche crossover films (e.g., atmospheric dark comedies, surreal dramas based on books) offers a differentiation strategy with proven audience interest.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 2. Patterns of Poor-Quality Movies — What to Avoid
# MAGIC
# MAGIC From **Q2**, the worst-rated movies tagged as *boring* or *overrated* have average ratings as low as 0.5 to 1.6. Titles such as *The Expedition*, *Water Boyy*, *Disaster Movie*, and *Andron* represent the lowest-rated films that audiences explicitly labelled negatively. Meanwhile, from **Q3**, films tagged *great acting* or *inspirational* — such as *The Shawshank Redemption* (avg ~4.41), *The Godfather* (avg ~4.32), and *12 Angry Men* (avg ~4.24) — consistently achieve ratings above 4.0.
# MAGIC
# MAGIC The contrast is clear: **strong performances and meaningful storytelling are the strongest predictors of high ratings**, while films perceived as unoriginal, dull, or overhyped receive the harshest audience judgement.
# MAGIC
# MAGIC **Recommendation:** The company should invest in strong screenwriting and casting as the primary quality drivers. Avoid greenlighting projects that rely heavily on spectacle or franchise recognition without substantive narrative depth, as these are most likely to be tagged *boring* or *overrated* by audiences.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 3. Tags That Signal Failure — Early Warning Indicators for Production
# MAGIC
# MAGIC From **Q5**, within the **1 to 2 rating range**, the most frequent tags are *boring*, *predictable*, *bad acting*, and *stupid* — all exceeding 200 occurrences. In the **2 to 3 range**, *boring* and *predictable* remain the dominant negative descriptors. This reveals that **predictability and lack of engagement are the two most consistent elements of poorly received movies** across all low-rating bands.
# MAGIC
# MAGIC Conversely, in the higher rating ranges (3 to 4, 4 to 5), positive descriptive tags related to genre or mood (e.g., *sci-fi*, *action*, *atmospheric*) dominate, suggesting that movies with **clear genre identity and strong tonal execution** are rewarded by audiences.
# MAGIC
# MAGIC **Recommendation:** During the development and pre-production phases, the company should implement audience-testing checkpoints that specifically screen for *predictability* and *engagement level*. Scripts and early cuts flagged as predictable or slow-paced should be revised before further investment. Additionally, ensuring films have a well-defined genre identity — rather than being generic or unfocused — will improve audience reception and ratings.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### Strategic Summary
# MAGIC
# MAGIC | Business Objective | Key Finding | Actionable Recommendation |
# MAGIC |---|---|---|
# MAGIC | Identify new genres to target | Audiences value *atmospheric*, *twist ending*, *visually appealing*, and *dark comedy* content beyond mainstream genres | Produce niche crossover films that combine these high-engagement attributes |
# MAGIC | Identify elements of bad movies to avoid | *Boring*, *predictable*, and *bad acting* are the strongest signals of poor reception | Invest in strong screenwriting and acting; implement predictability screening during development |
# MAGIC | Quality benchmarks | Top-rated tagged films average 4.0+ ratings; worst-rated average below 1.5 | Use audience tagging patterns as quality benchmarks during test screenings |
