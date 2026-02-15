"""
Video Presentation Script Generator
====================================
Generates a structured 7–10 minute video presentation script based on the
Jupyter Notebook analysis (Clifton Chen Yi - 231220B.ipynb), the dataset
(Course_Completion_Prediction.csv), and the accompanying Report.

The script reads the CSV to pull live summary statistics and outputs a
time-stamped, section-by-section narration covering:
  i.   Business objectives and methodology
  ii.  Jupyter Notebook walkthrough with dataset insights
  iii. Key findings and recommendations
  iv.  One challenge encountered and how it was addressed
  v.   Critical insights highlighted

Usage:
    python presentation_script.py            # prints to stdout
    python presentation_script.py -o FILE    # writes to FILE
"""

import argparse
import csv
import os
import sys

# ---------------------------------------------------------------------------
# Helper: load lightweight stats from the CSV without heavy dependencies
# ---------------------------------------------------------------------------

def _load_csv_stats(csv_path):
    """Return basic dataset stats read directly from the CSV."""
    stats = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        stats["num_columns"] = len(header)
        rows = list(reader)
        stats["num_rows"] = len(rows)

        # Target distribution
        try:
            target_idx = header.index("Completed")
        except ValueError:
            target_idx = None

        if target_idx is not None:
            completed = sum(
                1 for r in rows if r[target_idx].strip().lower() == "completed"
            )
            not_completed = stats["num_rows"] - completed
            stats["completed"] = completed
            stats["not_completed"] = not_completed
            stats["completed_pct"] = round(100 * completed / stats["num_rows"], 1)
            stats["not_completed_pct"] = round(
                100 * not_completed / stats["num_rows"], 1
            )
        else:
            stats["completed"] = stats["not_completed"] = 0
            stats["completed_pct"] = stats["not_completed_pct"] = 0.0

        stats["columns"] = header
    return stats


# ---------------------------------------------------------------------------
# Build the presentation script text
# ---------------------------------------------------------------------------

def build_script(stats):
    """Return the full narration script as a string."""
    lines = []

    def section(title, duration, body):
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"  {title}  [{duration}]")
        lines.append("=" * 72)
        lines.append("")
        lines.append(body.strip())
        lines.append("")

    # ── Title Slide ────────────────────────────────────────────────────────
    section(
        "SLIDE 1 – TITLE SLIDE",
        "~0:00 – 0:30",
        """\
[DELIVERY CUE: Speak clearly, make eye contact with camera, smile.]

NARRATION:
Hello, my name is Clifton Chen Yi, Admin Number 231220B.

Today I will present my Applied Machine-Learning project: "Predicting
Student Course Completion."  I will walk you through five areas:
  1. The business objectives and methodology.
  2. A hands-on walkthrough of my Jupyter Notebook with dataset insights.
  3. The key decisions I made and why.
  4. Key findings and actionable recommendations.
  5. A challenge I encountered and the lesson it taught me.

Let's get started.""",
    )

    # ── Business Objectives & Methodology ──────────────────────────────────
    section(
        "SLIDE 2 – BUSINESS OBJECTIVES & METHODOLOGY",
        "~0:30 – 2:30",
        f"""\
[DELIVERY CUE: Speak with conviction about why this problem matters.]

NARRATION:
The goal of this project is to predict whether a student will complete an
online course.  This is framed as a binary classification task — the target
variable has two values: "Completed" and "Not Completed."

Why does this matter?  Online learning platforms, especially MOOCs, face
dropout rates that often exceed 90 percent.  If we can identify at-risk
students early, platforms can trigger targeted interventions — personalised
reminders, additional support, or mentor outreach — and significantly
improve completion rates and revenue.

[SHOW SLIDE: Dataset summary table]

DATASET:
I used the "Student Course Completion Prediction Dataset" from Kaggle,
which contains {stats['num_rows']:,} student-course enrolment records
across {stats['num_columns']} features.  The features span demographics
(age, gender, education level), course metadata (duration, instructor
rating), engagement behaviour (login frequency, video completion rate,
quiz scores, progress percentage), and payment details.

The target variable is nearly balanced — roughly {stats['completed_pct']}%
Completed versus {stats['not_completed_pct']}% Not Completed — which means
accuracy is a valid metric and specialised class-imbalance techniques like
SMOTE were unnecessary.

[DELIVERY CUE: Pause briefly, then state criteria with confidence.]

SUCCESS CRITERIA — I set clear, measurable targets before building any
models:
  • Primary — F1-Score ≥ 0.70, because F1 balances precision and recall,
    which matters when both false positives (unnecessary interventions) and
    false negatives (missing at-risk students) carry real costs.
  • Secondary — ROC-AUC ≥ 0.75, Accuracy ≥ 0.70.
  • Generalisation — Cross-validation standard deviation < 0.02, to ensure
    stable performance across different data splits.

METHODOLOGY:
My approach followed a standard machine-learning pipeline:
  1. Import and explore the data (EDA).
  2. Clean, encode, and engineer features.
  3. Train three classifiers — Logistic Regression, Random Forest, and
     Gradient Boosting.
  4. Compare models on Accuracy, Precision, Recall, F1, and AUC.
  5. Tune the most promising model with GridSearchCV.
  6. Validate with Stratified 5-Fold Cross-Validation.

Each step involved deliberate decisions, which I will explain as we go.""",
    )

    # ── Notebook Walkthrough ───────────────────────────────────────────────
    section(
        "SLIDE 3 – NOTEBOOK WALKTHROUGH, DATASET INSIGHTS & KEY DECISIONS",
        "~2:30 – 5:30",
        f"""\
[DELIVERY CUE: Switch to screen-share of the notebook. Point to specific
cells as you narrate each section.]

NARRATION:
Let me now walk you through the key sections of my Jupyter Notebook,
highlighting the decisions I made and why.

[Show Notebook — Section 1 & 2: Imports & Data Loading]
After importing libraries such as pandas, scikit-learn, matplotlib, and
seaborn, I loaded the CSV file.  A quick .info() and .describe() confirmed
{stats['num_rows']:,} rows and {stats['num_columns']} columns with zero
missing values and no duplicates.

[Show Notebook — Section 3: Exploratory Data Analysis]
During EDA I examined distributions, correlations, and boxplots.  Here are
the insights that shaped my later decisions:

  • Age is roughly uniform between 17 and 52, mean around 26.
  • Video Completion Rate spans 0–100 %, showing high variability — this
    told me it would likely be a strong predictor.
  • Progress Percentage also spans the full range and showed the clearest
    separation between completers and non-completers in boxplots.
  • The correlation heatmap showed low inter-feature correlations (no pairs
    above 0.8), which confirmed that multicollinearity is not a concern
    and I could safely keep all features.
  • Critically, categorical features — Gender, Education Level, Employment
    Status — showed nearly uniform completion rates across categories.

[DELIVERY CUE: Emphasise the following insight confidently.]

  ★ KEY INSIGHT: This told me something important — demographics have
    limited predictive power for this problem.  Engagement behaviour is
    what truly differentiates completers from non-completers.  This finding
    guided all my subsequent feature engineering and modelling decisions.

[Show Notebook — Section 4: Data Cleaning & Feature Engineering]
Because the dataset was pre-cleaned, I artificially introduced missing
values and duplicates to practise real-world preprocessing.  Here are
the key decisions I made:

  DECISION — Median vs Mean Imputation:
    I chose median imputation because several features (Login_Frequency,
    Time_Spent_Hours) are right-skewed.  The median is more robust to
    outliers and better represents the typical value.  I also considered
    KNN imputation but rejected it — too computationally expensive for
    100,000 rows when only ~2 % of values were missing.

  DECISION — Hybrid Encoding Strategy:
    Rather than one-hot encoding everything (which would create a very
    wide feature matrix, especially for City with 15+ unique values), I
    used a targeted approach:
      – Ordinal encoding for ordered features (Education Level, Course
        Level, Internet Connection Quality) to preserve their natural rank.
      – One-hot encoding only for low-cardinality nominals (Gender, Device
        Type, Payment Mode).
      – Dropped high-cardinality identifiers (Student_ID, Name, City,
        Course_ID, Enrollment_Date) entirely to prevent overfitting and
        data leakage.

  DECISION — Feature Engineering Choices:
    I created two new features:
      – Assignment Completion Rate = Submitted / (Submitted + Missed)
        This ratio captures engagement quality better than raw counts.
      – Quiz Performance = Quiz Score Avg × Quiz Attempts
        This combines quality with effort — a student who scores 90 % on
        five quizzes shows stronger engagement than one who scores 90 % on
        just one.
    I also considered a Session-to-Login ratio but decided against it
    because it could produce extreme values for students with very low
    login frequency.

  • Scaled features with StandardScaler for Logistic Regression (essential
    for distance-based algorithms), while tree-based models used unscaled
    data since they are scale-invariant.

[Show Notebook — Section 5: Model Training]

  DECISION — Why These Three Models:
    I deliberately chose models from three different algorithm families:
      1. Logistic Regression — interpretable linear baseline.
      2. Random Forest — bagging ensemble of decision trees.
      3. Gradient Boosting — sequential boosting ensemble.
    I considered SVM but excluded it because its O(n²) complexity makes it
    impractical for a 100,000-row dataset without significant subsampling,
    which would reduce representativeness.""",
    )

    # ── Key Findings & Recommendations ─────────────────────────────────────
    section(
        "SLIDE 4 – KEY FINDINGS & RECOMMENDATIONS",
        "~5:30 – 8:00",
        """\
[DELIVERY CUE: Show the model comparison table or chart. Present results
with honesty — acknowledge targets were not met, but frame the learning.]

NARRATION:
Now let me share the key findings from the model comparison and evaluation.

MODEL PERFORMANCE:
[Show comparison table]
  • Logistic Regression — F1 = 0.5924, AUC = 0.6484, Accuracy = 0.6047
  • Random Forest       — F1 = 0.5778, AUC = 0.6288, Accuracy = 0.5933
  • Gradient Boosting   — F1 = 0.5932, AUC = 0.6441, Accuracy = 0.6040

I want to be transparent: all three models achieved around 60 % accuracy
and F1 ≈ 0.59, which falls short of my initial targets of F1 ≥ 0.70 and
AUC ≥ 0.75.  But this shortfall is itself a valuable finding — it tells us
that these 40 features simply do not contain enough signal to strongly
distinguish completers from non-completers.

[DELIVERY CUE: Lean in slightly — this is a surprising and important
result.]

  ★ KEY INSIGHT: Surprisingly, Logistic Regression — the simplest model —
    slightly outperformed both tree-based ensembles.  This tells us the
    predictive signal in this dataset is largely linear.  Adding model
    complexity did not help; it only added noise.

Random Forest showed severe overfitting: 100 % training accuracy but only
59.3 % on the test set.  I will discuss how I handled this in the
challenge section.

  DECISION — Why I Tuned Gradient Boosting (Not Logistic Regression):
    Although Logistic Regression performed marginally better, Gradient
    Boosting has more tunable hyperparameters (n_estimators, max_depth,
    learning_rate) and greater potential for improvement through tuning.
    I used GridSearchCV rather than RandomizedSearchCV because my
    parameter grid was small — only 8 combinations — making exhaustive
    search both feasible and thorough.

After tuning, the best parameters were learning_rate = 0.1, max_depth = 3,
and n_estimators = 100.  Notably, tuning selected a shallower tree depth
(3 instead of 5), confirming that simpler models generalise better when
the signal is weak.  However, test F1 dropped marginally from 0.5932 to
0.5886 — proving that hyperparameter tuning cannot create signal that does
not exist in the features.

VALIDATION:
[Show cross-validation results]
Stratified 5-Fold Cross-Validation confirmed stable generalisation:
  • All metrics had standard deviations < 0.011, meeting my < 0.02 target.
  • Cross-validation accuracy (~0.60) aligned with hold-out test accuracy,
    confirming that my single train-test split was reliable and the model
    is not overfitting to any particular data subset.

  ★ SCORECARD AGAINST SUCCESS CRITERIA:
      ✅ Cross-validation std < 0.02 — MET (all stds < 0.011)
      ❌ F1-Score ≥ 0.70 — NOT MET (best F1 = 0.5924)
      ❌ ROC-AUC ≥ 0.75 — NOT MET (best AUC = 0.6484)

FEATURE IMPORTANCE:
[Show feature importance chart]
The most important predictors were all engagement metrics:
  1. Progress Percentage
  2. Video Completion Rate
  3. Quiz Score Average
  4. Project Grade
  5. Assignment Completion Rate (my engineered feature)
Demographic features (Gender, Education Level, City) ranked lowest,
which directly confirms what EDA suggested: engagement data is far more
valuable than demographics for predicting course completion.

[DELIVERY CUE: Transition confidently to actionable recommendations.]

RECOMMENDATIONS FOR COURSE PROVIDERS:
  1. Monitor engagement metrics in real time — especially Progress
     Percentage and Video Completion Rate — to flag at-risk students early.
  2. Trigger automated interventions (reminder emails, mentor outreach)
     when predicted completion probability drops below a threshold.
  3. Collect richer behavioural data — forum participation, video watch
     patterns, prior course history — to push accuracy beyond 60 %.
  4. Use SHAP values in production for per-student explainability, so
     instructors understand why a student was flagged.
  5. Verify that Progress Percentage is available at prediction time to
     avoid data leakage in a live system.""",
    )

    # ── Challenge Encountered ──────────────────────────────────────────────
    section(
        "SLIDE 5 – CHALLENGE ENCOUNTERED & HOW IT WAS ADDRESSED",
        "~8:00 – 9:00",
        """\
[DELIVERY CUE: Be candid and reflective.  Acknowledging mistakes shows
maturity and genuine learning.]

NARRATION:
Let me share a significant challenge I encountered and the lesson it
taught me.

THE PROBLEM:
Random Forest achieved a perfect 100 % accuracy on the training set, yet
only 59.3 % on the test set.  At first, this was surprising — Random
Forest is generally considered a robust algorithm that resists overfitting
through bagging and averaging.

THE ROOT CAUSE:
I investigated and found that the default scikit-learn parameters place
no limit on tree depth and allow each leaf to contain as few as one
sample.  With 100,000 training rows and 40 features, the trees grew deep
enough to memorise every single training example — capturing noise rather
than genuine patterns.

HOW I ADDRESSED IT:
  1. I detected the overfitting by systematically comparing training
     versus test accuracy — a practice I now always perform.
  2. Rather than spending excessive time tuning Random Forest's many
     parameters, I made a strategic decision: I shifted focus to Gradient
     Boosting, which inherently controls complexity through its learning
     rate and already showed a much smaller train-test gap (63 % training
     vs 60 % test).
  3. During Gradient Boosting tuning, I explored shallower max_depth
     values (3 vs 5), confirming that simpler models generalise better
     when the underlying signal is weak.

THE LESSON:
[DELIVERY CUE: Speak slowly and deliberately for emphasis.]

This experience reinforced a critical lesson that I will carry into every
future project: a model that fits the training data perfectly is not a
good model — generalisation to unseen data is what truly matters.
Overfitting is not just a textbook concept; it has real consequences for
prediction quality.  Always compare training and test metrics before
trusting any model.""",
    )

    # ── Critical Insights & Closing ────────────────────────────────────────
    section(
        "SLIDE 6 – CRITICAL INSIGHTS & CLOSING",
        "~9:00 – 10:00",
        """\
[DELIVERY CUE: Summarise with energy and confidence.  These are the
takeaways the audience should remember.]

NARRATION:
To close, here are the five critical insights from this project — the
findings I believe are most important for anyone working on student
retention prediction.

  1. ENGAGEMENT OVER DEMOGRAPHICS
     Behavioural engagement metrics — Progress Percentage and Video
     Completion Rate — far outperform demographic features in predicting
     course completion.  Course providers should invest in tracking and
     leveraging these signals rather than relying on student profiles.

  2. SIMPLER MODELS CAN WIN
     Logistic Regression outperformed both Random Forest and Gradient
     Boosting.  This proves that when the underlying signal is linear,
     adding model complexity only adds noise.  Always start with a simple
     baseline before reaching for complex algorithms.

  3. KNOW YOUR DATA'S LIMITS
     Despite 40 features and 100,000 rows, the dataset offered limited
     discriminative power (AUC ≈ 0.65).  No amount of hyperparameter
     tuning can compensate for weak features.  The most impactful next
     step is collecting richer, real-time behavioural data such as forum
     participation and video watch patterns.

  4. WATCH FOR DATA LEAKAGE
     Progress Percentage — the single strongest predictor — may partially
     encode whether a student has already completed the course.  In
     production, it is essential to verify that every feature used at
     prediction time is available before the outcome is known.

  5. STABLE GENERALISATION MATTERS
     Cross-validation confirmed low variance (std < 0.011), meaning the
     model performs consistently across data splits.  Even though overall
     accuracy needs improvement, the model is stable enough to serve as a
     reliable baseline for deployment.

[DELIVERY CUE: End with a clear, confident closing.]

In summary, this project demonstrates a complete, rigorous machine-
learning workflow — from EDA through validation — and shows that
understanding your data's limitations is just as important as building
sophisticated models.  The decisions I made at each step were driven by
the data itself, not assumptions, and the challenges I encountered taught
me lessons that will make my future work stronger.

Thank you for watching.  I welcome any questions or feedback.""",
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a 7–10 minute video presentation script."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to write the script to (default: print to stdout).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to Course_Completion_Prediction.csv "
        "(default: same directory as this script).",
    )
    args = parser.parse_args()

    # Resolve CSV path
    if args.csv:
        csv_path = args.csv
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "Course_Completion_Prediction.csv")

    if not os.path.isfile(csv_path):
        sys.exit(f"Error: CSV file not found at {csv_path}")

    stats = _load_csv_stats(csv_path)
    script = build_script(stats)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(script)
        print(f"Presentation script written to {args.output}")
    else:
        print(script)


if __name__ == "__main__":
    main()
