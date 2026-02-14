# Databricks notebook source
# MAGIC %md
# MAGIC # IT2312 Big Data Modelling & Management
# MAGIC ## Individual Assignment - Big Data Processing – Movie Ratings
# MAGIC
# MAGIC This notebook performs data ingestion and exploration on the MovieLens 25M dataset using PySpark DataFrames.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1 – Data Ingestion

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Ingest the 3 files movies.csv, tags.csv and ratings.csv into DataBricks

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, lower, when

# Create SparkSession (already available in Databricks as 'spark')
# spark = SparkSession.builder.appName("IT2312_Assignment").getOrCreate()

# ---------- CONFIGURE YOUR FILE PATH HERE ----------
# Set BASE_PATH to the location where you uploaded movies.csv, tags.csv, and ratings.csv.
#
# Examples:
#   Unity Catalog Volume : "/Volumes/workspace/default/data/"
#   DBFS (if enabled)    : "/FileStore/tables/"
#
# You can also use the widget at the top of the notebook to change the path at runtime.
dbutils.widgets.text("base_path", "/FileStore/tables/", "Data folder path")
BASE_PATH = dbutils.widgets.get("base_path")
# Ensure the path ends with a slash
if not BASE_PATH.endswith("/"):
    BASE_PATH += "/"

movies_df = spark.read.csv(BASE_PATH + "movies.csv", header=True, inferSchema=True)
tags_df = spark.read.csv(BASE_PATH + "tags.csv", header=True, inferSchema=True)
ratings_df = spark.read.csv(BASE_PATH + "ratings.csv", header=True, inferSchema=True)

# Display the schemas
print("Movies Schema:")
movies_df.printSchema()

print("Tags Schema:")
tags_df.printSchema()

print("Ratings Schema:")
ratings_df.printSchema()

# COMMAND ----------

# Display sample data
print("Movies Sample:")
movies_df.show(5, truncate=False)

print("Tags Sample:")
tags_df.show(5, truncate=False)

print("Ratings Sample:")
ratings_df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Print the number of records and the number of columns for each data file

# COMMAND ----------

print(f"Movies  - Number of records: {movies_df.count()}, Number of columns: {len(movies_df.columns)}")
print(f"Tags    - Number of records: {tags_df.count()}, Number of columns: {len(tags_df.columns)}")
print(f"Ratings - Number of records: {ratings_df.count()}, Number of columns: {len(ratings_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2 – Data Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q1. Unique tags and their occurrence counts (excluding specified tags)
# MAGIC
# MAGIC Create a DataFrame showing the list of unique tags and the number of occurrences for each tag,
# MAGIC excluding: 'sci-fi', 'action', 'comedy', 'mystery', 'war', 'politics', 'religion', 'thriller'.
# MAGIC Sorted in descending order by tag occurrences.

# COMMAND ----------

# List of tags to exclude
exclude_tags = ['sci-fi', 'action', 'comedy', 'mystery', 'war', 'politics', 'religion', 'thriller']

# Create DataFrame with unique tags and their counts, excluding specified tags
q1_df = (
    tags_df
    .filter(~lower(col("tag")).isin([t.lower() for t in exclude_tags]))
    .groupBy("tag")
    .agg(count("*").alias("count"))
    .orderBy(col("count").desc())
)

q1_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q2. Movies tagged 'boring' or 'overrated' with title and average rating (top 10, ascending)
# MAGIC
# MAGIC Create a DataFrame that contains movies with the tags of 'boring' or 'overrated',
# MAGIC showing the title and average rating, sorted by average rating in ascending order.

# COMMAND ----------

# Filter tags for 'boring' or 'overrated'
boring_overrated_tags = tags_df.filter(lower(col("tag")).isin(["boring", "overrated"]))

# Calculate average rating per movie
avg_ratings = ratings_df.groupBy("movieId").agg(avg("rating").alias("avg_rating")).cache()

# Join tags with movies and average ratings
q2_df = (
    boring_overrated_tags
    .join(movies_df, on="movieId", how="inner")
    .join(avg_ratings, on="movieId", how="inner")
    .select("movieId", "title", "tag", "avg_rating")
    .orderBy(col("avg_rating").asc())
)

q2_df.show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q3. Movies tagged 'great acting' or 'inspirational' with title and average rating (top 10, descending)
# MAGIC
# MAGIC Create a DataFrame that contains movies with the tags of 'great acting' or 'inspirational',
# MAGIC showing the title and average rating, sorted by average rating in descending order.

# COMMAND ----------

# Filter tags for 'great acting' or 'inspirational'
great_inspirational_tags = tags_df.filter(lower(col("tag")).isin(["great acting", "inspirational"]))

# Join tags with movies and average ratings
q3_df = (
    great_inspirational_tags
    .join(movies_df, on="movieId", how="inner")
    .join(avg_ratings, on="movieId", how="inner")
    .select("movieId", "title", "tag", "avg_rating")
    .orderBy(col("avg_rating").desc())
)

q3_df.show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q4. Aggregate movie ratings into rating ranges
# MAGIC
# MAGIC Create a DataFrame that aggregates movie ratings into the ranges:
# MAGIC - Below 1: r < 1
# MAGIC - 1 to 2: r >= 1 and r < 2
# MAGIC - 2 to 3: r >= 2 and r < 3
# MAGIC - 3 to 4: r >= 3 and r < 4
# MAGIC - 4 to 5: r >= 4 and r < 5
# MAGIC - 5 and more: r >= 5
# MAGIC
# MAGIC Include columns: userId, movieId, rating, tag, rating_range.

# COMMAND ----------

# Join ratings with tags
ratings_tags_df = ratings_df.join(tags_df.select("userId", "movieId", "tag"), on=["userId", "movieId"], how="left")

# Create rating_range column using when/otherwise
q4_df = ratings_tags_df.withColumn(
    "rating_range",
    when(col("rating") < 1, "Below 1")
    .when((col("rating") >= 1) & (col("rating") < 2), "1 to 2")
    .when((col("rating") >= 2) & (col("rating") < 3), "2 to 3")
    .when((col("rating") >= 3) & (col("rating") < 4), "3 to 4")
    .when((col("rating") >= 4) & (col("rating") < 5), "4 to 5")
    .otherwise("5 and more")
).select("userId", "movieId", "rating", "tag", "rating_range")

q4_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q5. Aggregated rating ranges with tags and tag counts > 200
# MAGIC
# MAGIC Create a DataFrame showing aggregated movie rating ranges with their corresponding tags
# MAGIC and count of tags. Filter to show only tag counts > 200.
# MAGIC Sort by rating range ascending and tag counts descending.

# COMMAND ----------

# Filter out null tags, group by rating_range and tag, then filter counts > 200
q5_df = (
    q4_df
    .filter(col("tag").isNotNull())
    .groupBy("rating_range", "tag")
    .agg(count("*").alias("tag_count"))
    .filter(col("tag_count") > 200)
    .orderBy(col("rating_range").asc(), col("tag_count").desc())
)

q5_df.show(truncate=False)
