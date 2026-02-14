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
| `IT2312_Assignment.py` | Databricks-compatible Python notebook source |
| `IT2312_Assignment.ipynb` | Jupyter Notebook format |

## Setup Instructions

### Using Databricks

1. Download the MovieLens 25M dataset from https://grouplens.org/datasets/movielens/25m/
2. Upload `movies.csv`, `tags.csv`, and `ratings.csv` to Databricks FileStore (`/FileStore/tables/`)
3. Import `IT2312_Assignment.py` as a Databricks notebook, or upload `IT2312_Assignment.ipynb`
4. Run all cells in the notebook

### File Paths

The notebook expects data files at:
- `/FileStore/tables/movies.csv`
- `/FileStore/tables/tags.csv`
- `/FileStore/tables/ratings.csv`

Update these paths in the notebook if your files are stored in a different location.

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
