# Databricks notebook source

# MAGIC %md
# MAGIC # IT2312 2025 S2 Individual Assignment
# MAGIC ## Big Data Processing – Movie Ratings
# MAGIC
# MAGIC This notebook processes the MovieLens ml-25m dataset using PySpark DataFrames.
# MAGIC
# MAGIC **Dataset:** ml-25m (25,000,095 ratings and 1,093,360 tag applications across 62,423 movies)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1 – Data Ingestion
# MAGIC ### Task 1: Ingest movies.csv, tags.csv, and ratings.csv into Databricks

# COMMAND ----------

# Import required libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, lower, when, trim

# COMMAND ----------

# MAGIC %md
# MAGIC **Instructions:** Upload the ml-25m dataset files (movies.csv, tags.csv, ratings.csv) to Databricks DBFS.
# MAGIC
# MAGIC You can upload them via:
# MAGIC 1. Databricks UI: Data > Add Data > Upload File
# MAGIC 2. Or use the DBFS CLI
# MAGIC
# MAGIC Update the file paths below to match your upload location.

# COMMAND ----------

# Define file paths – update these to match your DBFS upload location
movies_path = "/FileStore/tables/movies.csv"
tags_path = "/FileStore/tables/tags.csv"
ratings_path = "/FileStore/tables/ratings.csv"

# COMMAND ----------

# Ingest movies.csv
movies_df = spark.read.csv(movies_path, header=True, inferSchema=True)

# Ingest tags.csv
tags_df = spark.read.csv(tags_path, header=True, inferSchema=True)

# Ingest ratings.csv
ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)

# COMMAND ----------

# Display sample data from each DataFrame
print("=== Movies DataFrame ===")
movies_df.show(5, truncate=False)

print("=== Tags DataFrame ===")
tags_df.show(5, truncate=False)

print("=== Ratings DataFrame ===")
ratings_df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2: Print the number of records and columns for each data file

# COMMAND ----------

# Print number of records and columns for each DataFrame
print(f"movies.csv  - Records: {movies_df.count()}, Columns: {len(movies_df.columns)}")
print(f"tags.csv    - Records: {tags_df.count()}, Columns: {len(tags_df.columns)}")
print(f"ratings.csv - Records: {ratings_df.count()}, Columns: {len(ratings_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 2 – Data Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1: Unique tags with occurrence counts (excluding specific tags)
# MAGIC
# MAGIC Create a DataFrame showing the list of unique tags and the number of occurrences for each tag,
# MAGIC excluding: `['sci-fi', 'action', 'comedy', 'mystery', 'war', 'politics', 'religion', 'thriller']`.
# MAGIC Sort in descending order by tag occurrences.

# COMMAND ----------

# List of tags to exclude (case-insensitive comparison)
exclude_tags = ['sci-fi', 'action', 'comedy', 'mystery', 'war', 'politics', 'religion', 'thriller']

# Create DataFrame with unique tags and their occurrence counts, excluding specified tags
task1_df = (
    tags_df
    .withColumn("tag_lower", lower(trim(col("tag"))))
    .filter(~col("tag_lower").isin(exclude_tags))
    .groupBy("tag")
    .agg(count("*").alias("count"))
    .orderBy(col("count").desc())
)

task1_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2: Movies tagged 'boring' or 'overrated' with average rating (Top 10, ascending)
# MAGIC
# MAGIC Create a DataFrame containing movies with the tags 'boring' or 'overrated',
# MAGIC showing the title and average rating. Sort by average rating ascending, display top 10.

# COMMAND ----------

# Filter tags for 'boring' or 'overrated' (case-insensitive)
boring_overrated_tags_df = (
    tags_df
    .withColumn("tag_lower", lower(trim(col("tag"))))
    .filter(col("tag_lower").isin(["boring", "overrated"]))
)

# Calculate average rating per movie
avg_ratings_df = (
    ratings_df
    .groupBy("movieId")
    .agg(avg("rating").alias("avg_rating"))
)

# Join tags with movies and average ratings
task2_df = (
    boring_overrated_tags_df
    .join(movies_df, on="movieId", how="inner")
    .join(avg_ratings_df, on="movieId", how="inner")
    .select("movieId", "title", "tag", "avg_rating")
    .orderBy(col("avg_rating").asc())
)

task2_df.show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3: Movies tagged 'great acting' or 'inspirational' with average rating (Top 10, descending)
# MAGIC
# MAGIC Create a DataFrame containing movies with the tags 'great acting' or 'inspirational',
# MAGIC showing the title and average rating. Sort by average rating descending, display top 10.

# COMMAND ----------

# Filter tags for 'great acting' or 'inspirational' (case-insensitive)
great_inspirational_tags_df = (
    tags_df
    .withColumn("tag_lower", lower(trim(col("tag"))))
    .filter(col("tag_lower").isin(["great acting", "inspirational"]))
)

# Join tags with movies and average ratings
task3_df = (
    great_inspirational_tags_df
    .join(movies_df, on="movieId", how="inner")
    .join(avg_ratings_df, on="movieId", how="inner")
    .select("movieId", "title", "tag", "avg_rating")
    .orderBy(col("avg_rating").desc())
)

task3_df.show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4: Aggregate movie ratings into rating ranges
# MAGIC
# MAGIC Create a DataFrame that categorizes ratings into ranges:
# MAGIC - Below 1: r < 1
# MAGIC - 1 to 2: r >= 1 and r < 2
# MAGIC - 2 to 3: r >= 2 and r < 3
# MAGIC - 3 to 4: r >= 3 and r < 4
# MAGIC - 4 to 5: r >= 4 and r < 5
# MAGIC - 5 and more: r >= 5
# MAGIC
# MAGIC Include columns: userId, movieId, rating, tag, rating_range

# COMMAND ----------

# Join ratings with tags
ratings_tags_df = ratings_df.join(tags_df, on=["userId", "movieId"], how="inner")

# Add rating_range column based on the rating value
task4_df = (
    ratings_tags_df
    .withColumn(
        "rating_range",
        when(col("rating") < 1, "Below 1")
        .when((col("rating") >= 1) & (col("rating") < 2), "1 to 2")
        .when((col("rating") >= 2) & (col("rating") < 3), "2 to 3")
        .when((col("rating") >= 3) & (col("rating") < 4), "3 to 4")
        .when((col("rating") >= 4) & (col("rating") < 5), "4 to 5")
        .otherwise("5 and more")
    )
    .select("userId", "movieId", "rating", "tag", "rating_range")
)

task4_df.show(20, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 5: Aggregated rating ranges with tags and tag counts > 200
# MAGIC
# MAGIC Create a DataFrame showing aggregated movie rating ranges with their corresponding tags
# MAGIC and tag counts. Filter to show only tag counts > 200.
# MAGIC Sort by rating range ascending and tag counts descending.

# COMMAND ----------

# Aggregate by rating_range and tag, count occurrences
task5_df = (
    task4_df
    .groupBy("rating_range", "tag")
    .agg(count("*").alias("tag_count"))
    .filter(col("tag_count") > 200)
    .orderBy(col("rating_range").asc(), col("tag_count").desc())
)

task5_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 6: Conclusions
# MAGIC
# MAGIC Based on the data exploration performed above, the following conclusions can be drawn:
# MAGIC
# MAGIC **1. Popular Genres Among Uncommon Tags:**
# MAGIC From Task 1, we can observe which tags are most frequently applied by users beyond the common genres
# MAGIC (sci-fi, action, comedy, mystery, war, politics, religion, thriller). This reveals niche interests and emerging
# MAGIC genres that the film production company could target. Tags with high occurrence counts represent
# MAGIC underserved audience interests that present production opportunities.
# MAGIC
# MAGIC **2. Characteristics of Poorly Rated vs. Highly Rated Movies:**
# MAGIC - Task 2 shows movies tagged as 'boring' or 'overrated' tend to have lower average ratings, confirming
# MAGIC   that audience perception aligns with these negative descriptors. The production company should study
# MAGIC   these titles to understand what elements lead to viewer dissatisfaction and avoid replicating them.
# MAGIC - Task 3 shows movies tagged as 'great acting' or 'inspirational' tend to receive higher average ratings,
# MAGIC   suggesting that investing in strong performances and meaningful storytelling correlates with audience approval.
# MAGIC
# MAGIC **3. Rating Distribution and Tag Patterns:**
# MAGIC - Task 4 and Task 5 reveal how tags distribute across different rating ranges. Tags that appear frequently
# MAGIC   in the higher rating ranges (4 to 5, 5 and more) indicate qualities associated with well-received films.
# MAGIC - Tags concentrated in lower rating ranges (Below 1, 1 to 2) highlight attributes the company should avoid.
# MAGIC - The filtering of tag counts > 200 ensures we focus on statistically significant patterns rather than outliers.
# MAGIC
# MAGIC **Actionable Recommendations:**
# MAGIC 1. Target production in genres/themes identified by high-frequency uncommon tags to capture underserved audiences.
# MAGIC 2. Avoid elements commonly found in 'boring' and 'overrated' movies (e.g., predictable plots, poor pacing).
# MAGIC 3. Invest in 'great acting' talent and 'inspirational' storytelling as these correlate strongly with higher ratings.
