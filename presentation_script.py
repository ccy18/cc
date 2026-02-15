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
NARRATION:
Hello, my name is Clifton Chen Yi, Admin Number 231220B.

In this presentation I will walk you through my Applied Machine-Learning
project: "Predicting Student Course Completion."  I will cover the business
objectives, the methodology, a walkthrough of my Jupyter Notebook with
dataset insights, the key findings and recommendations, and finally one
challenge I encountered and how I addressed it.

Let's begin.""",
    )

    # ── Business Objectives & Methodology ──────────────────────────────────
    section(
        "SLIDE 2 – BUSINESS OBJECTIVES & METHODOLOGY",
        "~0:30 – 2:30",
        f"""\
NARRATION:
The goal of this project is to predict whether a student will complete an
online course.  This is framed as a binary classification task — the target
variable has two values: "Completed" and "Not Completed."

Why does this matter?  Online learning platforms, especially MOOCs, face
dropout rates that often exceed 90 percent.  If we can identify at-risk
students early, platforms can trigger targeted interventions — personalised
reminders, additional support, or mentor outreach — and significantly
improve completion rates and revenue.

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

SUCCESS CRITERIA:
I defined three targets up front:
  • Primary — F1-Score ≥ 0.70
  • Secondary — ROC-AUC ≥ 0.75, Accuracy ≥ 0.70
  • Generalisation — Cross-validation standard deviation < 0.02

METHODOLOGY:
My approach followed a standard machine-learning pipeline:
  1. Import and explore the data (EDA).
  2. Clean, encode, and engineer features.
  3. Train three classifiers — Logistic Regression, Random Forest, and
     Gradient Boosting.
  4. Compare models on Accuracy, Precision, Recall, F1, and AUC.
  5. Tune the most promising model with GridSearchCV.
  6. Validate with Stratified 5-Fold Cross-Validation.""",
    )

    # ── Notebook Walkthrough ───────────────────────────────────────────────
    section(
        "SLIDE 3 – JUPYTER NOTEBOOK WALKTHROUGH & DATASET INSIGHTS",
        "~2:30 – 5:30",
        f"""\
NARRATION:
Let me now walk you through the key sections of my Jupyter Notebook.

[Show Notebook — Section 1 & 2: Imports & Data Loading]
After importing libraries such as pandas, scikit-learn, matplotlib, and
seaborn, I loaded the CSV file.  A quick .info() and .describe() confirmed
{stats['num_rows']:,} rows and {stats['num_columns']} columns with zero
missing values and no duplicates.

[Show Notebook — Section 3: Exploratory Data Analysis]
During EDA I examined distributions, correlations, and boxplots:
  • Age is roughly uniform between 17 and 52, mean around 26.
  • Video Completion Rate spans 0–100 %, showing high variability.
  • Progress Percentage also spans the full range — a strong separator.
  • The correlation heatmap showed low inter-feature correlations, meaning
    multicollinearity is not a concern.
  • Boxplots revealed that Progress Percentage and Video Completion Rate
    have the clearest separation between completers and non-completers.
  • Categorical features — Gender, Education Level, Employment Status —
    show nearly uniform completion rates, confirming that demographics have
    limited predictive power.

[Show Notebook — Section 4: Data Cleaning & Feature Engineering]
Because the dataset was pre-cleaned, I artificially introduced missing
values and duplicates to practise real-world preprocessing.  I then:
  • Applied median imputation (chosen over mean because several features
    are right-skewed).
  • Removed duplicates.
  • Dropped identifier columns (Student_ID, Name, City, Course_ID,
    Enrollment_Date) to prevent overfitting and data leakage.
  • Used a hybrid encoding strategy: ordinal encoding for ordered features
    (Education Level, Course Level, Internet Connection Quality) and
    one-hot encoding for low-cardinality nominals (Gender, Device Type,
    Payment Mode, etc.).
  • Engineered two new features:
      – Assignment Completion Rate = Submitted / (Submitted + Missed)
      – Quiz Performance = Quiz Score Avg × Quiz Attempts
  • Scaled features with StandardScaler for Logistic Regression.

[Show Notebook — Section 5: Model Training]
I trained three models:
  1. Logistic Regression — interpretable linear baseline.
  2. Random Forest — bagging ensemble of decision trees.
  3. Gradient Boosting — sequential boosting ensemble.
I deliberately excluded SVM because its O(n²) complexity makes it
impractical for a 100,000-row dataset.""",
    )

    # ── Key Findings & Recommendations ─────────────────────────────────────
    section(
        "SLIDE 4 – KEY FINDINGS & RECOMMENDATIONS",
        "~5:30 – 8:00",
        """\
NARRATION:
Here are the key findings from the model comparison and evaluation.

MODEL PERFORMANCE:
  • Logistic Regression — F1 = 0.5924, AUC = 0.6484, Accuracy = 0.6047
  • Random Forest       — F1 = 0.5778, AUC = 0.6288, Accuracy = 0.5933
  • Gradient Boosting   — F1 = 0.5932, AUC = 0.6441, Accuracy = 0.6040

All three models achieved around 60 % accuracy and F1 ≈ 0.59, falling
short of my initial targets of F1 ≥ 0.70 and AUC ≥ 0.75.

Surprisingly, Logistic Regression — the simplest model — slightly
outperformed both tree-based ensembles, suggesting that the predictive
signal in this dataset is largely linear.

Random Forest showed severe overfitting: 100 % training accuracy but only
59.3 % on the test set.

After tuning Gradient Boosting with GridSearchCV, the best parameters were
learning_rate = 0.1, max_depth = 3, and n_estimators = 100 — a shallower
tree depth was preferred.  However, tuning did not improve test performance
(F1 dropped marginally from 0.5932 to 0.5886), confirming that the
defaults were already near-optimal and that hyperparameter tuning cannot
create predictive signal that does not exist in the features.

VALIDATION:
Stratified 5-Fold Cross-Validation confirmed stable generalisation:
  • All metrics had standard deviations < 0.011, meeting my < 0.02 target.
  • Cross-validation accuracy (~0.60) aligned with hold-out test accuracy,
    confirming my single train-test split was reliable.

FEATURE IMPORTANCE:
The most important predictors were engagement metrics:
  1. Progress Percentage
  2. Video Completion Rate
  3. Quiz Score Average
  4. Project Grade
  5. Assignment Completion Rate (engineered)
Demographic features (Gender, Education Level, City) ranked lowest,
reinforcing that engagement data is far more valuable than demographics
for predicting course completion.

RECOMMENDATIONS:
  1. Monitor engagement metrics in real time to identify at-risk students
     early — especially Progress Percentage and Video Completion Rate.
  2. Trigger automated interventions (reminder emails, mentor outreach)
     when predicted completion probability drops below a threshold.
  3. Collect richer behavioural data — forum participation, video watch
     patterns, prior course history — to push accuracy beyond 60 %.
  4. Use SHAP values in production for per-student explainability.
  5. Verify that Progress Percentage is available at prediction time to
     avoid data leakage in a live system.""",
    )

    # ── Challenge Encountered ──────────────────────────────────────────────
    section(
        "SLIDE 5 – CHALLENGE ENCOUNTERED & HOW IT WAS ADDRESSED",
        "~8:00 – 9:00",
        """\
NARRATION:
One significant challenge I encountered was Random Forest's severe
overfitting.  The model achieved a perfect 100 % accuracy on the training
set, yet only 59.3 % on the test set.  At first, this was surprising
because Random Forest is generally considered robust.

What happened?  The default scikit-learn parameters place no limit on tree
depth and allow each leaf to contain as few as one sample.  With 100,000
training rows and 40 features, the trees grew deep enough to memorise every
training example — capturing noise rather than genuine patterns.

How I addressed it:
  • I recognised the overfitting by comparing training versus test accuracy.
  • Rather than spending excessive time tuning Random Forest, I shifted
    focus to Gradient Boosting, which inherently controls complexity
    through its learning rate and already showed a much smaller train-test
    gap (63 % training vs 60 % test).
  • During Gradient Boosting tuning, I explored shallower max_depth values
    (3 vs 5), confirming that simpler models generalise better when the
    underlying signal is weak.

This experience reinforced a critical lesson: a model that fits the
training data perfectly is not necessarily a good model — generalisation to
unseen data is what truly matters.""",
    )

    # ── Critical Insights & Closing ────────────────────────────────────────
    section(
        "SLIDE 6 – CRITICAL INSIGHTS & CLOSING",
        "~9:00 – 10:00",
        """\
NARRATION:
To wrap up, here are the critical insights from this project:

  1. ENGAGEMENT OVER DEMOGRAPHICS — Behavioural engagement metrics such as
     Progress Percentage and Video Completion Rate far outperform
     demographic features in predicting course completion.  Course
     providers should invest in tracking and leveraging these signals.

  2. SIMPLER MODELS CAN WIN — Logistic Regression outperformed more complex
     tree-based ensembles, proving that when the underlying signal is
     linear, added complexity only adds noise.

  3. KNOW YOUR DATA'S LIMITS — Despite 40 features and 100,000 rows, the
     dataset offered limited discriminative power (AUC ≈ 0.65).
     Hyperparameter tuning cannot compensate for weak features.  The most
     impactful next step is collecting richer, real-time behavioural data.

  4. WATCH FOR DATA LEAKAGE — Progress Percentage, the single strongest
     predictor, may partially encode whether a student has already
     completed the course.  In production, it is essential to verify that
     features are available at the time of prediction.

  5. STABLE GENERALISATION — Cross-validation confirmed low variance
     (std < 0.011), meaning the model performs consistently and is ready
     for deployment as a baseline, even if overall accuracy needs
     improvement.

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
