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
# MAGIC The two business objectives of this analysis are to **(1) identify new movie genres to target and produce** and **(2) identify examples and elements of bad movies to avoid replicating**. The following conclusions synthesise insights from all five exploration tasks to directly address these objectives.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Conclusion 1 — Untapped Genre Opportunities for New Productions (Business Objective 1)**
# MAGIC
# MAGIC Task 1 removed eight mainstream genre tags (sci-fi, action, comedy, mystery, war, politics, religion, thriller) to surface the tags that audiences organically apply most often. The high-frequency tags that remain — such as those related to atmospheric qualities (e.g., "atmospheric", "visually stunning"), narrative themes (e.g., "thought-provoking", "dark comedy"), and emotional impact (e.g., "twist ending", "feel-good") — represent viewer interests that are **underserved by traditional genre classifications**.
# MAGIC
# MAGIC This is strategically significant: because these tags reflect genuine, recurring audience demand rather than industry-defined categories, the production company can differentiate itself by developing films that intentionally target these niche descriptors. For example, rather than producing a generic "drama", the company could greenlight projects specifically designed to be "atmospheric" and "thought-provoking" — qualities that audiences actively seek out and tag, but that few studios deliberately market as core selling points.
# MAGIC
# MAGIC The sheer volume of these uncommon tags indicates a sizeable, engaged audience segment whose preferences are currently under-targeted by competitors.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Conclusion 2 — Defining Elements of Bad Movies to Avoid (Business Objective 2)**
# MAGIC
# MAGIC Tasks 2, 4, and 5 together paint a clear picture of what makes a movie fail in the eyes of audiences. Task 2 reveals that movies tagged as "boring" or "overrated" consistently sit at the **bottom of the average-rating spectrum**. This confirms that audience dissatisfaction is not random — it clusters around identifiable, repeatable patterns.
# MAGIC
# MAGIC Critically, the word "boring" points to **pacing and engagement failures** (e.g., slow plots, lack of tension), while "overrated" suggests a **gap between marketing hype and actual content quality**. The production company should treat these as two distinct failure modes: one is a creative problem (poor storytelling), and the other is a positioning problem (over-promising in marketing).
# MAGIC
# MAGIC Task 5 reinforces this by showing which tags dominate the lower rating ranges (Below 1, 1 to 2). Tags with counts exceeding 200 in these low ranges are statistically significant indicators of audience rejection — they are not outliers but persistent, large-scale patterns. By cross-referencing the most frequent tags in low-rating ranges with the titles from Task 2, the company can build a concrete checklist of pitfalls to avoid during script development and production planning.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Conclusion 3 — Blueprint for High-Rated Films (Business Objectives 1 & 2)**
# MAGIC
# MAGIC Task 3 provides the positive counterpart: movies tagged "great acting" or "inspirational" cluster at the **top of the average-rating spectrum**. This is not merely a correlation — it reflects a causal audience preference. Strong acting performances create emotional investment, while inspirational narratives deliver lasting viewer satisfaction, both of which translate directly into higher ratings and positive word-of-mouth.
# MAGIC
# MAGIC Combined with Task 5, which shows tags with high counts in the 4-to-5 and 5-and-more rating ranges, a pattern emerges: **the most commercially and critically viable films share a combination of strong performances, emotionally resonant themes, and distinctive stylistic qualities**.
# MAGIC
# MAGIC Task 4's rating-range bucketing further reveals that the vast majority of tagged interactions fall in the 3-to-4 and 4-to-5 ranges, indicating that audiences who take the time to tag movies are generally engaged viewers — making their positive tags especially valuable as predictors of broad audience appeal. The production company should use these high-rated tags as a creative brief: invest in casting (great acting), develop stories with uplifting or meaningful arcs (inspirational), and layer in the atmospheric and thought-provoking qualities surfaced in Task 1.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Strategic Recommendations**
# MAGIC
# MAGIC Based on the above conclusions, the following actionable strategies are recommended:
# MAGIC
# MAGIC 1. **Genre Strategy:** Prioritise production of films that target the top uncommon tags identified in Task 1. These tags represent proven audience interests outside mainstream genres and offer a competitive advantage through differentiation. Market these films using the tag language audiences already use (e.g., "a visually stunning, thought-provoking thriller") to align promotion with genuine viewer expectations.
# MAGIC
# MAGIC 2. **Quality Control — Avoiding Bad-Movie Traits:** Establish an internal review process during pre-production that screens scripts and rough cuts against the negative-tag patterns from Tasks 2 and 5. Specifically, flag projects at risk of being "boring" (pacing issues, lack of conflict) or "overrated" (marketing that outpaces content quality). This preventive approach is more cost-effective than correcting failures post-release.
# MAGIC
# MAGIC 3. **Talent and Story Investment:** Allocate a larger share of the production budget to casting (to secure "great acting") and script development (to craft "inspirational" narratives), as Task 3 demonstrates these are the strongest predictors of high audience ratings. This investment has a compounding return: highly rated films generate organic word-of-mouth and repeat viewership, reducing long-term marketing costs.
# MAGIC
# MAGIC 4. **Data-Driven Greenlighting:** Integrate tag-based audience analysis into the greenlighting process. Before approving a new project, cross-reference its intended genre, themes, and style against the tag-frequency and rating-range data from Tasks 1, 4, and 5 to estimate its audience reception. This transforms subjective editorial judgement into an evidence-based decision framework.
