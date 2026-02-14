# IT2312 Big Data Modelling & Management - Individual Assignment

## Big Data Processing – Movie Ratings

This repository contains the PySpark notebook for the IT2312 Individual Assignment on processing and analyzing the MovieLens 25M dataset.

## Dataset

The assignment uses the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/) which contains:
- **25,000,095 ratings** and **1,093,360 tag applications** across **62,423 movies**
- Created by **162,541 users** between January 09, 1995 and November 21, 2019

### Required Files
- `movies.csv` - Movie information (movieId, title, genres)
- `tags.csv` - User-applied tags (userId, movieId, tag, timestamp)
- `ratings.csv` - User ratings (userId, movieId, rating, timestamp)

## Files

The three notebook files below are **the same notebook** in different formats. You only need to import **one** of them into Databricks.

| File | Description |
|------|-------------|
| `IT2312_Assignment.dbc` | Databricks Archive — **recommended**, import directly via Databricks Workspace |
| `IT2312_Assignment.py` | Databricks Python notebook source — alternative import option |
| `IT2312_Assignment.ipynb` | Jupyter Notebook format — alternative import option |

> **You do NOT need to import all 3 files.** Pick one (`.dbc` is recommended) and import it into Databricks.

## How to Import into Databricks

Import **one** of the three notebook files using any of the options below:

### Option 1: Import the `.dbc` archive (recommended)

1. In Databricks, go to your **Workspace**
2. Right-click on your target folder → **Import**
3. Select **File** and upload `IT2312_Assignment.dbc`
4. The notebook will appear ready to use

### Option 2: Import the `.py` file

1. In Databricks, go to your **Workspace**
2. Right-click on your target folder → **Import**
3. Select **File** and upload `IT2312_Assignment.py`
4. Databricks will recognise it as a notebook source

### Option 3: Import the `.ipynb` file

1. In Databricks, go to your **Workspace**
2. Right-click on your target folder → **Import**
3. Select **File** and upload `IT2312_Assignment.ipynb`

### After importing — upload the dataset

1. Download the MovieLens 25M dataset from https://grouplens.org/datasets/movielens/25m/
2. Extract the downloaded zip file to get the CSV files
3. Upload `movies.csv`, `tags.csv`, and `ratings.csv` to a **Unity Catalog Volume** by following these steps:

#### Step-by-step: Create a Volume and upload files

1. In the left sidebar, click **Catalog**
2. Under "My organization", click **`workspace`**
3. Click the **`default`** schema
4. Click **Create Volume** (or click the **+** button next to Volumes)
5. Name it **`my_volume`** and click **Create**
6. Click on the newly created **`my_volume`** volume
7. Click **Upload to this volume** and upload these 3 files:
   - `movies.csv`
   - `tags.csv`
   - `ratings.csv`

> Your files will now be at `/Volumes/workspace/default/my_volume/` — this matches the notebook's default path, so you can run the notebook without changing anything.

#### Run the notebook

1. Go back to **Workspace** and open the imported notebook
2. A **widget text box** labelled "Data folder path" will appear at the top — it should already show `/Volumes/workspace/default/my_volume/`
3. Click **Run All** to execute all cells

### File Paths

The notebook uses a configurable `BASE_PATH` widget (default: `/Volumes/workspace/default/my_volume/`).

If you followed the steps above and named your volume `my_volume`, the default path will work — no changes needed.

If you used a different catalog, schema, or volume name, update the widget at the top of the notebook to: `/Volumes/<catalog>/<schema>/<volume>/`

The widget appears at the top of the notebook when you run the first code cell.

## Assignment Structure

### Part 1 – Data Ingestion (10 marks)
1. Ingest the 3 CSV files into DataBricks
2. Print the number of records and columns for each file

### Part 2 – Data Exploration (80 marks)
1. **Q1**: Unique tags with occurrence counts (excluding specified genre tags), sorted descending
2. **Q2**: Movies tagged 'boring' or 'overrated' with average rating, top 10 ascending
3. **Q3**: Movies tagged 'great acting' or 'inspirational' with average rating, top 10 descending
4. **Q4**: Rating ranges aggregation (Below 1, 1-2, 2-3, 3-4, 4-5, 5+)
5. **Q5**: Rating ranges with tags and tag counts >200, sorted by range and count
