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

| File | Description |
|------|-------------|
| `IT2312_Assignment.dbc` | Databricks Archive — import directly via Databricks Workspace |
| `IT2312_Assignment.py` | Databricks Python notebook source — importable as a notebook |
| `IT2312_Assignment.ipynb` | Jupyter Notebook format — importable into Databricks or Jupyter |

## How to Import into Databricks

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
2. Upload `movies.csv`, `tags.csv`, and `ratings.csv` to Databricks:
   - **Using the UI**: In your Workspace, click **File** > **Upload data to DBFS** and upload the three CSV files to `/FileStore/tables/`
   - **Using Unity Catalog Volumes**: Navigate to **Catalog** > select your catalog and schema > **Create Volume**, then upload the CSV files there
3. Open the imported notebook — a **widget text box** will appear at the top labelled "Data folder path"
4. Enter the path where you uploaded the files (e.g. `/FileStore/tables/` or `/Volumes/workspace/default/data/`)
5. Run all cells in the notebook

### File Paths

The notebook uses a configurable `BASE_PATH` widget. Set it to wherever you uploaded the CSV files:

| Upload method | Path to enter |
|---------------|---------------|
| DBFS upload (UI) | `/FileStore/tables/` |
| Unity Catalog Volume | `/Volumes/<catalog>/<schema>/<volume>/` |

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
