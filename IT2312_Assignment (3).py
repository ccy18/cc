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
movies_df = spark.read.csv("/Volumes/workspace/default/my_volume/movies.csv.bz2", header=True, inferSchema=True)

# Read tags.csv
tags_df = spark.read.csv("/Volumes/workspace/default/my_volume/tags.csv.bz2", header=True, inferSchema=True)

# Read ratings.csv
ratings_df = spark.read.csv("/Volumes/workspace/default/my_volume/ratings.csv.bz2", header=True, inferSchema=True)

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
# MAGIC ### Q4. Aggregate movie ratings into ranges with userId, movieId, rating, tag, and rating_range columns

# COMMAND ----------

from pyspark.sql.functions import when

# Join ratings with tags
ratings_tags_df = ratings_df.join(tags_df, on=["userId", "movieId"], how="inner")

# Add rating_range column and select required columns
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
