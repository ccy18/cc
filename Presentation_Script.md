# IT2312 2025 S2 Individual Assignment — Presentation Script

> **Target duration:** 7–8 minutes (within the 5–10 minute requirement)
>
> **How to use this script:** Read the quoted text aloud. Follow the stage directions in **bold brackets** — they tell you what to show on screen, where to gesture, and how to pace your delivery. Rehearse at least twice so you can speak naturally rather than reading word-for-word.

---

## SECTION 1 — Introduction (≈ 40 seconds)

**[Show the notebook title cell on screen. Sit upright, look at the camera, and smile briefly before you begin.]**

> "Hi, I'm [YOUR NAME], and welcome to my presentation for the IT2312 Individual Assignment — Big Data Processing with Movie Ratings."

**[Pause 1 second, then continue at a confident, steady pace.]**

> "Here's the scenario. I'm a data scientist at a film production company, and leadership has set two clear business objectives."

**[Raise one finger.]**

> "First — identify new movie genres we should target and produce."

**[Raise a second finger.]**

> "And second — identify the elements of bad movies so we can avoid repeating those mistakes."

**[Gesture toward the screen.]**

> "To tackle these objectives, I used the MovieLens ml-25m dataset — that's over 25 million ratings and about 1 million user-generated tags spanning more than 62,000 movies. Everything you'll see was built in Databricks using PySpark DataFrames — no SQL. Let me show you how."

---

## SECTION 2 — Part 1: Data Ingestion (≈ 50 seconds)

**[Scroll to the Part 1 cells. Point at the code as you speak.]**

> "Starting with Part 1 — Data Ingestion. I uploaded three CSV files into Databricks: movies, tags, and ratings."

**[Highlight or hover over the `spark.read.csv` lines.]**

> "Each file is loaded with `spark.read.csv`, using `header=True` and `inferSchema=True` — so Spark automatically picks up the column names and data types."

**[Scroll to the `show()` output. Tap or gesture at the table.]**

> "Here's a quick preview of each DataFrame — you can see the columns and sample rows confirming everything loaded correctly."

**[Scroll to the record-count output. Slow down slightly to let the numbers land.]**

> "And here are the counts:
> - movies — 62,423 records, 3 columns.
> - tags — 1,093,360 records, 4 columns.
> - ratings — 25,000,095 records, 4 columns.
>
> So the full dataset is accounted for. Let's move into the exploration."

---

## SECTION 3 — Part 2, Task 1: Unique Tag Occurrences (≈ 1 minute)

**[Scroll to Part 2, Task 1 code. Gesture at the exclusion list in the code.]**

> "Now, Task 1 asks: what are the most popular tags outside the obvious genre labels?"

**[Pause briefly — let the question register.]**

> "I excluded eight common genre tags — sci-fi, action, comedy, mystery, war, politics, religion, and thriller — because we already know those exist. What's more interesting is what audiences tag *beyond* those."

**[Point to the code logic.]**

> "I normalised each tag to lowercase using `lower` and `trim`, filtered out the exclusion list with `isin` and the tilde operator, then grouped by tag and counted occurrences."

**[Scroll to the output table. Use the Databricks bar chart visualisation if available — click the chart icon below the table. Otherwise point at top rows.]**

> "Look at the results. Tags like 'atmospheric', 'thought-provoking', and 'visually stunning' rank at the top. These aren't traditional genres — they're *qualities* that audiences actively seek out and label. And that's exactly the kind of insight our first business objective needs: these tags tell the company *what to produce next*."

---

## SECTION 4 — Part 2, Task 2: Boring / Overrated Movies (≈ 50 seconds)

**[Scroll to Task 2. Adopt a slightly more serious tone.]**

> "Task 2 flips the lens — now we're looking at the *worst* movies. Specifically, movies tagged 'boring' or 'overrated'."

**[Point to the filter and join code.]**

> "I filtered tags for those two words, calculated the average rating per movie, joined with the movies table for titles, and sorted ascending — so the lowest-rated appear first."

**[Scroll to the top-10 output. Point at the low avg_rating values.]**

> "Notice how low these ratings are. The data confirms that when audiences call a movie 'boring', it really *is* poorly rated — this isn't just subjective complaining, it's a consistent pattern across millions of ratings."

**[Look at the camera for emphasis.]**

> "'Boring' signals a *pacing* problem — the movie fails to hold attention. 'Overrated' signals a *positioning* problem — the marketing promised more than the film delivered. Both are avoidable, and that's critical for our second business objective."

---

## SECTION 5 — Part 2, Task 3: Great Acting / Inspirational Movies (≈ 50 seconds)

**[Scroll to Task 3. Shift to a brighter, more positive tone.]**

> "Now the positive side — Task 3 looks at movies tagged 'great acting' or 'inspirational'."

**[Gesture at the output table.]**

> "Same approach: filter, join, and sort — this time descending, so the *highest*-rated movies appear first."

**[Point at the high avg_rating values.]**

> "And look at these ratings — they're dramatically higher than what we saw in Task 2."

**[Look at the camera.]**

> "This tells us something important: strong acting builds emotional connection, and inspirational stories leave audiences feeling satisfied. Both of those qualities translate directly into high ratings — and high ratings drive word-of-mouth, which is the most cost-effective marketing there is."

**[Brief transition.]**

> "So already, from Tasks 2 and 3, a clear contrast emerges: the worst movies bore or mislead audiences; the best ones invest in performances and meaningful stories."

---

## SECTION 6 — Part 2, Task 4: Rating Range Bucketing (≈ 45 seconds)

**[Scroll to Task 4 code.]**

> "Task 4 takes a different angle. Instead of looking at individual movies, I bucketed every rating into ranges — Below 1, 1 to 2, 2 to 3, 3 to 4, 4 to 5, and 5 and more."

**[Point to the `when` chain in the code.]**

> "I joined ratings with tags on both userId and movieId, then used PySpark's `when` function to assign each rating to a `rating_range` bucket."

**[Scroll to the output table.]**

> "The output includes userId, movieId, rating, tag, and the new rating_range column. This bucketed view is the foundation for the next task — it lets us see *which tags cluster in which quality tier*."

---

## SECTION 7 — Part 2, Task 5: Rating Ranges × Tags, Counts > 200 (≈ 50 seconds)

**[Scroll to Task 5 code and output. If Databricks chart is available, switch to a grouped bar chart.]**

> "Task 5 is where the picture really comes together. I grouped by rating_range and tag, counted the occurrences, and filtered for tag counts above 200 — so we only see statistically significant patterns."

**[Point at the output, highlighting the contrast between low and high ranges.]**

> "This table is powerful. Look at which tags dominate the 4-to-5 and 5-and-more ranges versus the Below-1 and 1-to-2 ranges. The highest-rated tiers are dominated by positive descriptors; the lowest tiers by negative ones."

**[Look at the camera.]**

> "This cross-tabulation is the evidence base for everything I'll conclude next — it shows us *exactly* which audience-defined qualities correlate with success and failure."

---

## SECTION 8 — Conclusions and Recommendations (≈ 2 minutes)

**[Scroll to the Conclusions cell. Sit up straight, adopt a deliberate pace — this is the most important section.]**

> "Let me now bring it all together with three conclusions tied directly to our business objectives."

**[Pause 1 second.]**

> "**Conclusion one — untapped genre opportunities.**"

**[Gesture toward the screen.]**

> "Task 1 showed us that audiences consistently tag movies with descriptors like 'atmospheric', 'thought-provoking', and 'visually stunning' — and these aren't traditional genres. They represent *real demand* that falls outside what most studios deliberately target. The recommendation is clear: the company should develop films that intentionally embody these qualities and market them using the exact language audiences already use. That's how you capture an underserved segment."

**[Transition.]**

> "**Conclusion two — what makes a bad movie.**"

**[Slightly slower pace for emphasis.]**

> "Tasks 2 and 5 revealed two distinct failure patterns. Movies called 'boring' suffer from *pacing and engagement problems* — slow plots, lack of tension. Movies called 'overrated' suffer from a *perception gap* — the marketing over-promised and the content under-delivered. These are two different problems requiring two different fixes: one in the creative process, the other in the marketing strategy. Both should be screened for *before* a film reaches the audience."

**[Transition.]**

> "**Conclusion three — a blueprint for success.**"

**[Lean slightly forward for conviction.]**

> "Task 3 proved that 'great acting' and 'inspirational' are the strongest predictors of high ratings. Task 5 confirmed that positive-quality tags dominate the highest rating ranges. So the blueprint is this: invest in talented actors, develop emotionally meaningful stories, and layer in the distinctive stylistic qualities — like 'atmospheric' and 'thought-provoking' — that Task 1 told us audiences are actively looking for."

**[Pause, then list recommendations with a counting gesture.]**

> "Based on all of this, I have four strategic recommendations."

> "One — **target uncommon-tag genres**. Build films around the high-frequency niche descriptors from Task 1 and market them using that same audience language."

> "Two — **implement quality screening**. During pre-production, evaluate scripts and cuts for 'boring' or 'overrated' risk factors identified in Tasks 2 and 5."

> "Three — **invest in talent and story**. Allocate more budget to casting and script development — Task 3 shows these yield the highest audience returns."

> "Four — **adopt data-driven greenlighting**. Use tag-frequency and rating-range data to evaluate new projects *before* committing resources — turning gut instinct into evidence-based decisions."

---

## SECTION 9 — Closing (≈ 15 seconds)

**[Look directly at the camera. Speak with calm confidence.]**

> "To sum up: the data tells us that audiences reward strong acting, meaningful storytelling, and distinctive style — and they punish boredom and hype. Armed with these insights, the production company can make smarter, data-driven decisions about what to create and what to avoid. Thank you for watching."

**[Hold eye contact for 2 seconds, then stop recording.]**

---

## Presenter Notes — Delivery Coaching

### Timing
- **Total estimated time:** 7–8 minutes at a moderate speaking pace (≈ 140 wpm).
- Rehearse with a timer. If you finish under 5 minutes, slow down and add brief pauses after key points. If over 10, trim the code explanations slightly.

### Screen Setup
- **Share your Databricks notebook** full-screen throughout. Scroll to the relevant cell as you discuss each task.
- **Use Databricks chart visualisations** for Tasks 1 and 5: click the bar-chart icon below each output table to create a visual. This satisfies the rubric's requirement for "use of visuals and graphics."
- Consider briefly **highlighting or hovering** over key code lines (e.g., the `filter`, `groupBy`, `when` chains) as you explain them.

### Confidence and Body Language
- **Posture:** Sit upright with shoulders relaxed. Avoid leaning too far forward or slouching.
- **Eye contact:** Look at the camera lens (not the screen) for at least 2–3 seconds at a time, especially when making key points and during conclusions. Glance at the screen only when pointing at specific data.
- **Gestures:** Use hand gestures naturally — count on fingers when listing items, point at data on screen, open palms when explaining concepts. Avoid fidgeting.
- **Facial expression:** Smile briefly at the start and end. Show genuine interest when discussing findings.

### Voice and Fluency
- **Pace:** Speak at a moderate, steady pace. Deliberately slow down for important conclusions and speed up slightly for routine code descriptions.
- **Emphasis:** Stress key words (marked with *italics* or **bold** in the script). For example, say "these are *real* audience preferences" with audible emphasis on "real."
- **Pauses:** Pause for 1–2 seconds after each conclusion heading and before transitioning between sections. Pauses project confidence and give the audience time to absorb.
- **Avoid filler words:** If you catch yourself about to say "um" or "uh", pause silently instead — silence sounds more professional.

### Engaging the Audience
- The script includes **rhetorical questions** (e.g., "what are the most popular tags outside the obvious genre labels?") — deliver these with a slightly raised tone to signal a question, then pause before answering.
- Use **signposting transitions** like "Let me now bring it all together" and "Now the positive side" to keep the viewer oriented.
- **Conviction:** When presenting conclusions and recommendations, speak with certainty. Phrases like "the data confirms", "the recommendation is clear", and "the blueprint is this" are designed to sound decisive — deliver them that way.

### Before You Record
1. Close unnecessary browser tabs and notifications.
2. Test your microphone and webcam — ensure clear audio and good lighting.
3. Have the notebook pre-scrolled to the title cell so you're ready to begin.
4. Keep this script open in a separate window or printed out, but avoid reading it verbatim — use it as a guide and speak naturally.
