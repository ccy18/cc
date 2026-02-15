# IT2312 2025 S2 Individual Assignment — Presentation Script

> **Target duration:** 5–10 minutes
> **Tip:** Speak at a comfortable pace. Share your Databricks notebook on-screen and scroll through each cell as you discuss it.

---

## SLIDE / SECTION 1 — Introduction (≈ 45 seconds)

**[Show the notebook title cell on screen]**

> "Hello, my name is [YOUR NAME] and this is my presentation for the IT2312 Individual Assignment.
>
> I play the role of a data scientist at a film production company with two business objectives: first, identify new movie genres to target and produce; and second, identify elements of bad movies to avoid replicating.
>
> I am using the MovieLens ml-25m dataset — over 25 million ratings and about 1.09 million tag applications across more than 62,000 movies. All analysis was done in Databricks using PySpark DataFrames. Let me walk you through my work."

---

## SLIDE / SECTION 2 — Part 1: Data Ingestion (≈ 1 minute)

**[Scroll to the Part 1 cells — file paths, spark.read.csv, and the count output]**

> "For Part 1, I uploaded movies.csv, tags.csv, and ratings.csv into Databricks DBFS, then loaded each file using `spark.read.csv` with `header=True` and `inferSchema=True` so Spark automatically detects column names and data types.
>
> **[Point to the show() output]**
>
> Here is a sample of each DataFrame confirming the data loaded correctly.
>
> **[Scroll to the record-count cell]**
>
> For Task 2, the record and column counts are:
>
> - movies.csv: 62,423 records, 3 columns — movieId, title, genres.
> - tags.csv: 1,093,360 records, 4 columns — userId, movieId, tag, timestamp.
> - ratings.csv: 25,000,095 records, 4 columns — userId, movieId, rating, timestamp.
>
> This confirms the dataset was ingested completely."

---

## SLIDE / SECTION 3 — Part 2, Task 1: Unique Tag Occurrences (≈ 1 minute)

**[Scroll to the Part 2 Task 1 code and output]**

> "In Task 1, I found the unique tags and their occurrence counts, excluding eight common genre tags: sci-fi, action, comedy, mystery, war, politics, religion, and thriller.
>
> I converted each tag to lowercase using `lower` and `trim` for case-insensitive comparison, filtered out the exclusion list using `isin` negated with the tilde operator, then grouped by tag and counted occurrences, sorting in descending order.
>
> **[Point to the output table]**
>
> The result shows popular tags outside those mainstream genres — things like 'atmospheric', 'visually stunning', and 'thought-provoking'. These represent genuine audience interests beyond traditional genre labels and are directly relevant to our first business objective of identifying new genres to target."

---

## SLIDE / SECTION 4 — Part 2, Task 2: Boring / Overrated Movies (≈ 1 minute)

**[Scroll to the Task 2 code and output]**

> "Task 2 looks at movies tagged 'boring' or 'overrated' — the kind of movies the company wants to avoid.
>
> I filtered the tags for those two values case-insensitively, calculated the average rating per movie using `groupBy` and `avg`, then joined with the movies DataFrame to get titles. The result is sorted by average rating ascending, showing the top 10.
>
> **[Point to the output table]**
>
> Movies tagged 'boring' or 'overrated' consistently have low average ratings. 'Boring' suggests pacing and engagement failures, while 'overrated' suggests a gap between marketing hype and actual quality."

---

## SLIDE / SECTION 5 — Part 2, Task 3: Great Acting / Inspirational Movies (≈ 1 minute)

**[Scroll to the Task 3 code and output]**

> "Task 3 is the positive counterpart — movies tagged 'great acting' or 'inspirational', joined with average ratings and sorted descending, top 10.
>
> **[Point to the output table]**
>
> These movies cluster at the top of the rating spectrum. Strong acting creates emotional investment and inspirational stories deliver lasting satisfaction — both translate directly into high ratings. Comparing Tasks 2 and 3, the contrast is clear: the worst movies lack engagement, while the best deliver strong performances and meaningful narratives."

---

## SLIDE / SECTION 6 — Part 2, Task 4: Rating Range Bucketing (≈ 1 minute)

**[Scroll to the Task 4 code and output]**

> "In Task 4, I categorised individual ratings into defined ranges: Below 1, 1 to 2, 2 to 3, 3 to 4, 4 to 5, and 5 and more.
>
> I joined ratings with tags on userId and movieId, then used PySpark's `when` function to create a `rating_range` column. The output includes userId, movieId, rating, tag, and rating_range.
>
> **[Point to the output table]**
>
> This bucketed view lets us analyse how tags distribute across different quality tiers — which tags appear most in the lowest versus highest rating ranges."

---

## SLIDE / SECTION 7 — Part 2, Task 5: Rating Ranges with Tag Counts > 200 (≈ 1 minute)

**[Scroll to the Task 5 code and output]**

> "Task 5 builds on Task 4. I grouped by rating_range and tag, counted occurrences, and filtered for tag counts greater than 200, sorting by rating range ascending and tag count descending.
>
> **[Point to the output table]**
>
> By keeping only counts above 200, we focus on statistically significant patterns. We can see which tags dominate each rating tier — tags frequent in the 4-to-5 and 5-and-more ranges tell us what audiences value most, while tags concentrated in lower ranges highlight attributes associated with viewer rejection. This cross-tabulation forms the foundation for my conclusions."

---

## SLIDE / SECTION 8 — Conclusions and Recommendations (≈ 2 minutes)

**[Scroll to the Task 6 Conclusions cell]**

> "Now for the conclusions, addressing our two business objectives.
>
> **My first conclusion** is about genre opportunity. Task 1 surfaced high-frequency tags like 'atmospheric', 'thought-provoking', and 'visually stunning' that fall outside mainstream genre labels. These represent untapped audience demand. The company can differentiate by deliberately targeting these descriptors — for example, greenlighting a project designed to be 'atmospheric and thought-provoking' rather than a generic drama.
>
> **My second conclusion** addresses bad-movie traits. Tasks 2 and 5 show that 'boring' and 'overrated' movies sit at the bottom of the rating spectrum, and certain negative tags consistently dominate the lowest rating ranges. 'Boring' points to pacing failures; 'overrated' points to a marketing-versus-quality gap. These are two distinct failure modes the company should guard against during pre-production.
>
> **My third conclusion** ties it together. Task 3 shows 'great acting' and 'inspirational' are the strongest predictors of high ratings. Combined with the tag-frequency data from Tasks 1 and 5, a blueprint emerges: the most successful films combine strong performances, emotionally resonant stories, and distinctive stylistic qualities.
>
> From these conclusions, I have four recommendations:
>
> 1. **Genre strategy** — target films around the top uncommon tags and market them using the language audiences already use.
> 2. **Quality control** — screen scripts against negative-tag patterns to catch 'boring' or 'overrated' traits before release.
> 3. **Talent investment** — prioritise casting and script development, since these drive the highest ratings.
> 4. **Data-driven greenlighting** — integrate tag and rating-range analysis into project approvals for evidence-based decisions."

---

## SLIDE / SECTION 9 — Closing (≈ 15 seconds)

> "That concludes my presentation. The data shows audiences value strong acting, meaningful stories, and unique stylistic qualities — and penalise movies that are boring or over-hyped. By using these insights, the production company can make smarter decisions about what to produce and what to avoid. Thank you for watching."

---

## Presenter Notes

- **Total estimated time:** approximately 7–8 minutes at a moderate speaking pace.
- **Screen sharing:** keep the Databricks notebook visible throughout and scroll to the relevant cell as you discuss each task.
- **Eye contact:** look at the camera periodically rather than reading from the script word-for-word.
- **Visuals:** if time permits, use Databricks' built-in chart visualisations for Tasks 1 and 5 to make the output more engaging.
- **Practice:** rehearse at least once to ensure you stay within the 5–10 minute window.
