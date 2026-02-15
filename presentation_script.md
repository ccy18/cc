# IT2312 2025 S2 Individual Assignment — Presentation Script

**Duration:** 8–10 minutes  
**Format:** Screen-share of Databricks notebook with webcam overlay  

---

## 🎬 Pre-Recording Checklist

Before you press record, complete every item below:

- [ ] **Webcam on** — position yourself in the top-right corner overlay so the audience can see your face throughout.
- [ ] **Notebook ready** — run all cells in the Databricks notebook so every output is already visible; collapse any code cells you plan to expand live.
- [ ] **Zoom level** — set your browser to 110–125 % so table outputs are legible on a recorded screen.
- [ ] **Mouse cursor** — use a large or highlighted cursor (Databricks Accessibility settings or an OS pointer-size increase) so viewers can follow where you point.
- [ ] **Quiet environment** — close other apps, silence notifications.
- [ ] **Water nearby** — keep a glass of water within reach for a mid-presentation sip if needed.
- [ ] **Practice run** — rehearse at least once end-to-end out loud, timing yourself. Aim for 8–9 minutes.

---

## 🗣️ Delivery Tips (apply throughout)

| Rubric Area | What to Do |
|---|---|
| **Eye contact** | Look at your webcam lens (not the screen) when delivering key points and conclusions. Glance at the notebook only when pointing at data. |
| **Posture** | Sit upright with shoulders relaxed. Avoid slouching or swaying. |
| **Gestures** | Use hand gestures when listing items ("first… second… third…"). Keep gestures within the webcam frame. |
| **Vocal variety** | Slow down and lower your pitch for important findings. Speed up slightly for code walkthroughs to signal familiarity. Avoid monotone. |
| **Pauses** | Pause for 1–2 seconds after stating a key finding — this lets the audience absorb the point and makes you sound confident. |
| **Engagement** | Use rhetorical questions ("So what does this tell us?") and bridging phrases ("Now, here is where it gets interesting…") to keep the audience invested. |
| **Fluency** | If you stumble, don't restart — just pause, breathe, and continue. Short pauses sound more natural than filler words like "um" or "uh". |

---

## SECTION 1 — Introduction (~1 minute)

**🎥 Visual:** Show the notebook title cell. Look at the webcam.

> Hello, and welcome to my presentation for the IT2312 Big Data Modelling and Management Individual Assignment.
>
> *[Look at webcam, speak with energy]*
>
> Imagine you are running a film production company. You need to decide: what kind of movies should we make next — and just as importantly, what mistakes should we avoid? Those are exactly the two business objectives I was tasked with:
>
> *[Raise one finger]* First, **identify new movie genres to target and produce**.  
> *[Raise a second finger]* Second, **identify examples and elements of bad movies to avoid replicating**.
>
> To answer these questions, I used the **MovieLens ml-25m dataset** — one of the largest publicly available movie-rating datasets. It contains roughly **25 million ratings** and over **1 million user-applied tags** across more than **62,000 movies**, contributed by about **162,000 users** between 1995 and 2019.
>
> I processed all of this data using **PySpark DataFrames** on the **Databricks** platform — no SQL, purely DataFrame operations.
>
> Let me walk you through how I did it.

---

## SECTION 2 — Part 1: Data Ingestion (~1 minute)

**🎥 Visual:** Scroll to the data-ingestion code cells. Use your cursor to highlight the `spark.read.csv` lines.

> Starting with **Part 1 — Data Ingestion**.
>
> *[Point cursor at the three `spark.read.csv` lines]*
>
> I loaded three CSV files into Databricks using `spark.read.csv` with `header=True` and `inferSchema=True` so that column names and data types are automatically detected.
>
> The three files are:
> - **movies.csv** — movie ID, title, and genres.
> - **tags.csv** — user-applied free-text tags with timestamps.
> - **ratings.csv** — user ratings on a 0.5-to-5 star scale with timestamps.
>
> *[Scroll to the print output — hover cursor over each line as you read]*
>
> After loading, I printed the record and column counts:
>
> - **movies.csv**: **62,423 records**, **3 columns**.
> - **tags.csv**: **1,093,360 records**, **4 columns**.
> - **ratings.csv**: **25,000,095 records**, **4 columns**. That is 25 million rows — a genuinely large dataset.
>
> *[Pause 1 second, look at webcam]*
>
> So the data is loaded and verified. Let's move on to exploring it.

---

## SECTION 3 — Q1: Unique Tags Excluding Common Genres (~1.5 minutes)

**🎥 Visual:** Scroll to Q1. Briefly highlight the `exclude_tags` list in the code, then scroll to the output table.

> Now we enter **Part 2 — Data Exploration**. Question 1 asks: what are the unique tags and how often do they appear, **after excluding** the common genre tags — sci-fi, action, comedy, mystery, war, politics, religion, and thriller?
>
> *[Point cursor at the exclude_tags list]*
>
> Here is the list of tags I filtered out. I used `lower()` for case-insensitive matching, then grouped by tag and counted occurrences, sorted in descending order.
>
> *[Scroll to the Q1 output table — move cursor down the rows as you speak]*
>
> And here are the results. Notice something interesting — the top tags are **not genre labels**. They are *experiential qualities*:
>
> - **atmospheric** — over 6,500 occurrences  
> - **surreal** — over 5,300  
> - **based on a book** — over 5,000  
> - **twist ending** — about 4,800
>
> We also see **funny**, **visually appealing**, **dystopia**, and **dark comedy** ranking highly.
>
> *[Look at webcam, pause]*
>
> So what does this tell us? It tells us that when audiences tag a movie, they describe **how it made them feel** — the atmosphere, the surprise, the visual impact — rather than just its genre. This is a key insight, and I will connect it to our business recommendations in the conclusion.

---

## SECTION 4 — Q2: Boring or Overrated Movies (~1.5 minutes)

**🎥 Visual:** Scroll to Q2. Highlight the `filter` line, then scroll to the output table.

> **Question 2** asks: which movies are tagged as **boring** or **overrated**, and what are their average ratings? I need to show the top 10, sorted by average rating in **ascending order** — worst first.
>
> *[Point cursor at the filter and join logic]*
>
> I filtered the tags for "boring" or "overrated", computed the average rating per movie from the 25 million ratings, joined with movie titles, and sorted ascending.
>
> *[Scroll to the Q2 output table — hover over the top rows]*
>
> Here are the worst-rated movies that audiences explicitly called out:
>
> - **The Expedition** and **Water Boyy** — average rating of just **0.5**. That is the absolute bottom.
> - **Disaster Movie** — about **1.2**. The title is almost self-explanatory.
> - **Andron** — roughly **1.45**.
> - Further down: **When Do We Eat**, **Arson Mom**, **The Aftermath** — all in the **1.5 to 1.6** range.
>
> *[Look at webcam with a slight head-shake for emphasis]*
>
> These are not just low-rated movies — they are movies that audiences felt strongly enough about to actively tag as boring or overrated. They give us real, data-backed examples of what *not* to replicate.
>
> Now let's look at the opposite end of the spectrum.

---

## SECTION 5 — Q3: Great Acting or Inspirational Movies (~1.5 minutes)

**🎥 Visual:** Scroll to Q3. Scroll to the output table.

> **Question 3** is the positive counterpart. Here I look at movies tagged **great acting** or **inspirational**, sorted by average rating in **descending order** — best first.
>
> *[Scroll to the Q3 output table — hover over each title as you mention it]*
>
> And the results read like a hall of fame:
>
> - **The Shawshank Redemption** — average rating approximately **4.41**
> - **The Godfather** — approximately **4.32**
> - **The Usual Suspects** — approximately **4.28**
> - **The Godfather Part II** — approximately **4.26**
> - **12 Angry Men** — approximately **4.24**
> - **Fight Club** — approximately **4.00**
>
> *[Pause. Look at webcam.]*
>
> Now here is where it gets really interesting. Compare these numbers with Q2: the best-rated films average **above 4.0**. The worst-rated average **below 1.6**. That is a **massive gap** — nearly 3 full stars.
>
> The pattern is unmistakable: **strong acting and meaningful storytelling** are the strongest predictors of whether audiences love or hate a film. This directly informs our second business objective — what elements to invest in and what to avoid.

---

## SECTION 6 — Q4: Rating Range Aggregation (~1 minute)

**🎥 Visual:** Scroll to Q4. Highlight the `when` chain in the code, then scroll to the output.

> **Question 4** prepares the data for deeper analysis. I need to categorise every rating into a named range and combine it with user tags.
>
> *[Point cursor at the `when` chain]*
>
> First, I performed an **inner join** between the ratings and tags DataFrames on userId and movieId. Then I used PySpark's `when` function — essentially a series of if-else conditions — to create a new column called **rating_range**:
>
> - **Below 1** for ratings under 1  
> - **1 to 2**, **2 to 3**, **3 to 4**, **4 to 5** for each subsequent band  
> - **5 and more** for perfect scores
>
> *[Scroll to the Q4 output — point at the rating_range column]*
>
> Each row now links a user, a movie, the original rating, the tag they applied, and the rating band it falls into. This gives us a structured view that sets up the final analysis in Q5.

---

## SECTION 7 — Q5: Tag Counts by Rating Range (~1 minute)

**🎥 Visual:** Scroll to Q5. Scroll to the output table. Use cursor to trace down the rows.

> **Question 5** is where the real story emerges. I grouped by **rating_range** and **tag**, counted occurrences, kept only tags with **more than 200 appearances**, and sorted by rating range ascending, then tag count descending.
>
> *[Scroll to Q5 output — trace cursor along the rows as you speak]*
>
> Look at the **1 to 2 rating range** — the lowest band with significant data:
>
> - **boring** — **452** occurrences  
> - **predictable** — **275**  
> - **bad acting** — **228**  
> - **stupid** — **211**
>
> These four tags are the DNA of a terrible movie.
>
> *[Scroll down to the higher ranges]*
>
> Now contrast this with the **3 to 4** and **4 to 5** ranges. Here, the dominant tags are **sci-fi**, **action**, **atmospheric** — genre and mood descriptors, not complaints.
>
> *[Look at webcam, pause]*
>
> The takeaway? Low-rated movies are defined by **quality failures** — boring, predictable, bad acting. High-rated movies are defined by **genre identity and atmosphere**. This is one of the most actionable insights in the entire analysis.

---

## SECTION 8 — Q6: Conclusions and Recommendations (~2 minutes)

**🎥 Visual:** Scroll to the Q6 conclusions markdown. Point at each heading as you discuss it. At the end, highlight the Strategic Summary table.

> Now for the most important part — **Question 6, my conclusions**. I have drawn three conclusions that directly address our company's business objectives.
>
> *[Point at Conclusion 1 heading]*
>
> **Conclusion 1: Opportunity in Uncommon Genres.**
>
> From Q1, we learned that audiences' most-used tags — atmospheric, surreal, twist ending, visually appealing, dark comedy — describe *experiences*, not genres. This means there is strong demand for films that deliver distinctive storytelling qualities beyond the mainstream. My recommendation: produce **niche crossover films** that combine multiple high-frequency attributes — for example, *a visually striking dystopian drama with a twist ending*. This lets us differentiate rather than compete head-on in saturated genres.
>
> *[Point at Conclusion 2 heading]*
>
> **Conclusion 2: What Makes Movies Fail — and Succeed.**
>
> From Q2 and Q3, the contrast is stark. Boring and overrated films score as low as 0.5. Films with great acting and inspirational storytelling consistently score above 4.0. The data is clear: **screenwriting and casting are the primary quality drivers**. My recommendation: prioritise investment in strong scripts and talented casts, and avoid greenlighting projects that rely purely on spectacle without substance.
>
> *[Point at Conclusion 3 heading]*
>
> **Conclusion 3: Early Warning Tags — Screen for Predictability.**
>
> From Q5, the tags boring, predictable, bad acting, and stupid are overwhelmingly concentrated in the lowest rating band. Predictability and disengagement are the two most consistent markers of failure. My recommendation: implement **audience-testing checkpoints** during development. If early screenings flag a script or cut as predictable, revise it before committing further budget.
>
> *[Scroll to the Strategic Summary table — point at each row]*
>
> I have summarised all three conclusions in this strategic table, mapping each business objective to its key finding and actionable recommendation.
>
> *[Pause 2 seconds. Look at webcam.]*

---

## SECTION 9 — Closing (~30 seconds)

**🎥 Visual:** Stay on the Strategic Summary table or scroll back to the notebook title. Look at the webcam for the entire closing.

> *[Speak slowly and clearly — this is the last impression]*
>
> To wrap up: through this big data analysis of **25 million ratings** using PySpark on Databricks, I have shown that:
>
> *[Count on fingers]*
>
> **One** — audiences crave atmospheric, surprising, and visually distinctive content beyond mainstream genres, which presents clear opportunities for niche production.
>
> **Two** — boring and predictable are the strongest negative signals; great acting and inspiration drive the highest ratings.
>
> **Three** — these patterns translate into actionable recommendations the company can implement immediately — from genre targeting to quality screening during development.
>
> *[Smile, nod confidently]*
>
> Thank you for watching. This has been my presentation for the IT2312 Individual Assignment.

---

## 📋 Post-Recording Checklist

- [ ] Watch your recording end-to-end. Check that all table outputs are legible.
- [ ] Verify the video is between **5 and 10 minutes** in length.
- [ ] Confirm your face is visible throughout (webcam overlay).
- [ ] Ensure audio is clear with no background noise.
- [ ] Export/save in a standard format (MP4 recommended) and upload to BrightSpace.

---

*End of script — Target duration: 8–10 minutes.*
