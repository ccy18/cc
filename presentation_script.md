# IT2312 2025 S2 Individual Assignment — Presentation Script

**Duration:** 8–10 minutes  
**Format:** Screen-share of Databricks notebook with narration  
**Tip:** Open each cell's output before you start recording so you can scroll through results smoothly.

---

## SLIDE / SECTION 1 — Introduction (~1 minute)

> Hello, and welcome to my presentation for the IT2312 Big Data Modelling and Management Individual Assignment.
>
> In this assignment I am playing the role of a data scientist working for a film production company. The company has tasked me with two business objectives:
>
> 1. **Identify new movie genres to target and produce**, and  
> 2. **Identify examples and elements of bad movies to avoid replicating.**
>
> To achieve this, I am using the **MovieLens ml-25m dataset**, which contains approximately **25 million ratings** and over **1 million tag applications** across more than **62,000 movies**, created by roughly **162,000 users** between 1995 and 2019.
>
> I processed this data using **PySpark DataFrames** on the **Databricks** platform. Let me now walk you through my notebook.

---

## SLIDE / SECTION 2 — Part 1: Data Ingestion (~1 minute)

> Starting with **Part 1 – Data Ingestion**.
>
> *(Scroll to the data ingestion cells)*
>
> I ingested three CSV files into Databricks using `spark.read.csv` with `header=True` and `inferSchema=True`:
>
> - **movies.csv** — contains the movie ID, title, and genres.
> - **tags.csv** — contains user-applied free-text tags for movies, along with timestamps.
> - **ratings.csv** — contains user ratings on a 5-star scale with timestamps.
>
> *(Scroll to the record-count output)*
>
> After loading, I printed the number of records and columns for each file:
>
> - **movies.csv** has **62,423 records** and **3 columns** — movieId, title, and genres.
> - **tags.csv** has **1,093,360 records** and **4 columns** — userId, movieId, tag, and timestamp.
> - **ratings.csv** has **25,000,095 records** and **4 columns** — userId, movieId, rating, and timestamp.
>
> This confirms the data was ingested successfully and matches the expected dataset size.

---

## SLIDE / SECTION 3 — Q1: Unique Tags Excluding Common Genres (~1.5 minutes)

> Moving on to **Part 2 – Data Exploration**, starting with **Question 1**.
>
> *(Scroll to the Q1 code cell)*
>
> The task is to create a DataFrame showing unique tags and their occurrence counts, **excluding** the following mainstream genre-related tags: sci-fi, action, comedy, mystery, war, politics, religion, and thriller.
>
> I used `lower()` for case-insensitive filtering, then grouped by the tag column and counted occurrences using `count`. Finally, I sorted in **descending order** by count.
>
> *(Scroll to the Q1 output table)*
>
> Looking at the results, the top tags are descriptive qualities rather than genres:
>
> - **atmospheric** — over 6,500 occurrences
> - **surreal** — over 5,300 occurrences
> - **based on a book** — over 5,000 occurrences
> - **twist ending** — about 4,800 occurrences
> - **funny**, **visually appealing**, **dystopia**, and **dark comedy** also appear prominently.
>
> This tells us that audiences actively tag movies with **experiential qualities** — things like atmosphere, narrative surprises, and visual style — rather than just broad genre labels. This insight is very valuable for our business objectives, which I will discuss in the conclusion.

---

## SLIDE / SECTION 4 — Q2: Boring or Overrated Movies (~1.5 minutes)

> **Question 2** asks me to find movies tagged as **boring** or **overrated**, show their title and average rating, and display the **top 10 sorted in ascending order** by average rating.
>
> *(Scroll to the Q2 code cell)*
>
> I filtered the tags DataFrame for tags matching "boring" or "overrated" using case-insensitive comparison. I then computed the **average rating per movie** from the ratings DataFrame, joined the results with the movies DataFrame to get titles, and sorted by average rating in ascending order.
>
> *(Scroll to the Q2 output table)*
>
> The results show the **worst-rated movies** that audiences explicitly labelled negatively:
>
> - **The Expedition** and **Water Boyy** have the lowest average ratings at **0.5** — effectively the worst possible score.
> - **Disaster Movie** at roughly **1.2** and **Andron** at about **1.45** follow closely.
> - Other titles like **When Do We Eat**, **Arson Mom**, and **The Aftermath** also fall in the **1.5–1.6 range**.
>
> These are films that audiences not only rated poorly but also felt strongly enough about to tag as boring or overrated. This gives us concrete examples of what to avoid in future productions.

---

## SLIDE / SECTION 5 — Q3: Great Acting or Inspirational Movies (~1.5 minutes)

> **Question 3** is the positive counterpart — movies tagged as **great acting** or **inspirational**, sorted by average rating in **descending order**, top 10.
>
> *(Scroll to the Q3 code cell)*
>
> The approach is similar to Q2: I filtered for the positive tags, joined with average ratings and movie titles, and sorted descending.
>
> *(Scroll to the Q3 output table)*
>
> The results include some of the most acclaimed films in cinema history:
>
> - **The Shawshank Redemption** — average rating of approximately **4.41**
> - **The Godfather** — approximately **4.32**
> - **The Usual Suspects** — approximately **4.28**
> - **The Godfather Part II** — approximately **4.26**
> - **12 Angry Men** — approximately **4.24**
> - **Fight Club** — approximately **4.00**
>
> The contrast with Q2 is striking. The best-rated films average **above 4.0**, while the worst-rated average **below 1.6**. This confirms that **strong acting and meaningful storytelling** are the strongest predictors of audience satisfaction.

---

## SLIDE / SECTION 6 — Q4: Rating Range Aggregation (~1 minute)

> **Question 4** asks me to categorise each rating into ranges and create a DataFrame with the columns userId, movieId, rating, tag, and a new column called **rating_range**.
>
> *(Scroll to the Q4 code cell)*
>
> I first performed an **inner join** between the ratings and tags DataFrames on userId and movieId — this links each rating to its corresponding tags. Then I used PySpark's `when` function to classify each rating:
>
> - **Below 1** for ratings less than 1
> - **1 to 2** for ratings from 1 up to but not including 2
> - **2 to 3**, **3 to 4**, and **4 to 5** following the same pattern
> - **5 and more** for ratings of 5 or above
>
> *(Scroll to the Q4 output table)*
>
> As you can see, each row now shows the user, the movie, the original rating, the tag they applied, and the corresponding rating range. This structured data forms the foundation for Q5.

---

## SLIDE / SECTION 7 — Q5: Tag Counts by Rating Range (~1 minute)

> **Question 5** builds on Q4 by aggregating the data — grouping by **rating_range** and **tag**, counting the number of tag occurrences, filtering for counts **greater than 200**, and sorting by rating range ascending and tag count descending.
>
> *(Scroll to the Q5 code cell and output)*
>
> The results reveal important patterns:
>
> - In the **1 to 2 rating range**, the top tags are **boring** with 452 occurrences, **predictable** with 275, **bad acting** with 228, and **stupid** with 211. These are the hallmarks of terrible movies.
> - In the **2 to 3 range**, **boring** and **predictable** remain dominant, reinforcing that these are the most consistent negative signals.
> - Moving up to the **3 to 4** and **4 to 5 ranges**, we see positive genre and mood tags like **sci-fi**, **action**, and **atmospheric** dominating — indicating that well-received films have a **clear genre identity**.
>
> This is a powerful insight: the tags that appear most in low-rated bands are quality-related complaints, while high-rated bands feature genre and thematic descriptors.

---

## SLIDE / SECTION 8 — Q6: Conclusions and Recommendations (~2 minutes)

> Finally, **Question 6 — my conclusions**. I have drawn three key conclusions that directly address our business objectives.
>
> *(Scroll to the Q6 markdown section)*
>
> **Conclusion 1: Opportunity in Uncommon Genres.**
>
> From Q1, we discovered that after removing mainstream genre tags, the most popular tags are experiential qualities — atmospheric, surreal, twist ending, visually appealing, dark comedy. This shows audiences actively seek films with distinctive storytelling elements beyond standard genres. My recommendation is to produce **niche crossover films** that combine these attributes — for example, a visually appealing dystopian drama with a twist ending. This offers a differentiation strategy in a market saturated with mainstream genre films.
>
> **Conclusion 2: Patterns of Poor-Quality Movies — What to Avoid.**
>
> From Q2 and Q3, we see a stark contrast: boring and overrated films score as low as 0.5, while great acting and inspirational films consistently score above 4.0. The data clearly shows that **strong screenwriting and casting are the primary drivers of audience satisfaction**. My recommendation is to invest in quality storytelling and avoid greenlighting projects that rely on spectacle or franchise recognition without narrative substance.
>
> **Conclusion 3: Tags That Signal Failure — Early Warning Indicators.**
>
> From Q5, the tags boring, predictable, bad acting, and stupid are overwhelmingly concentrated in the 1-to-2 rating range. Predictability and lack of engagement are the most consistent elements of failure. My recommendation is to implement **audience-testing checkpoints** during development — specifically screening for predictability and pacing issues. Scripts flagged as predictable should be revised before further investment.
>
> *(Point to the Strategic Summary table)*
>
> I have summarised these findings in a strategic table mapping each business objective to the key finding and the corresponding actionable recommendation.

---

## SLIDE / SECTION 9 — Closing (~30 seconds)

> In summary, through this big data analysis of 25 million ratings using PySpark on Databricks, I have:
>
> 1. Identified that audiences value **atmospheric, surprising, and visually distinctive content** beyond mainstream genres — presenting opportunities for niche film production.
> 2. Demonstrated that **boring and predictable** are the strongest negative signals, while **great acting and inspiration** drive the highest ratings.
> 3. Provided actionable recommendations for the production company to **target underserved audience preferences** and **avoid the pitfalls** that lead to poor reception.
>
> Thank you for watching. This has been my presentation for the IT2312 Individual Assignment.

---

*End of script — Total estimated duration: 8–10 minutes.*
