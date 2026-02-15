# IT2312 2025 S2 Individual Assignment - Course Completion Prediction
# Name: Clifton Chen Yi
# Admin Number: 231220B

# ## Name: Clifton Chen Yi

# ## Admin Number: 231220B

# ## Brief Overview (provide your video link here too)
# 
# **Problem Statement:** Predict whether a student will complete an online course, framed as a **binary classification** task (Completed vs Not Completed).
# 
# **Motivation & Real-World Relevance:** Online learning platforms face high dropout rates — often exceeding 90% on MOOCs ([Onah et al., 2014](https://doi.org/10.13140/RG.2.1.2402.0009)). Early identification of at-risk students enables targeted interventions (e.g., personalised reminders, additional support) that can significantly improve course completion rates and platform revenue.
# 
# **Dataset:** [Student Course Completion Prediction Dataset](https://www.kaggle.com/datasets/nisargpatel344/student-course-completion-prediction-dataset) — 100,000 student-course enrolment records with 40 features covering demographics, course metadata, engagement behaviour, and payment details.
# 
# **Success Criteria:**
# - **Primary metric:** F1-Score ≥ 0.70 — chosen as a balanced metric for classification that weighs both false positives (unnecessary interventions) and false negatives (missing at-risk students).
# - **Secondary metrics:** ROC-AUC ≥ 0.75 and Accuracy ≥ 0.70 — to validate discriminative ability and overall correctness.
# - **Generalisation:** Cross-validation standard deviation < 0.02, indicating stable performance across data splits.
# 
# **Approach:** Train and compare three supervised classifiers — Logistic Regression (interpretable baseline), Random Forest (non-linear ensemble), and Gradient Boosting (sequential boosting) — then tune the best performer using GridSearchCV and validate with Stratified K-Fold Cross-Validation.
# 
# **Video link:** *(insert link here)*

# <a id='table_of_contents'></a>
# 
# 1. [Import libraries](#imports)
# 2. [Import data](#import_data)
# 3. [Data exploration](#data_exploration)
# 4. [Data cleaning and preparation](#data_cleaning)
# 5. [Model training](#model_training)<br>
# 6. [Model comparison](#model_comparsion)<br>
# 7. [Tuning](#tuning)<br>
# 8. [Validation](#validation)<br>
# 9. [Conclusion](#conclusion)<br>

# # 1. Import libraries <a id='imports'></a>
# [Back to top](#table_of_contents)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, GridSearchCV, StratifiedKFold,
                                     cross_val_score, StratifiedShuffleSplit)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score, roc_curve)

import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully.")

# # 2. Import data <a id='import_data'></a>
# [Back to top](#table_of_contents)

df = pd.read_csv('Course_Completion_Prediction.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
df.head()

print("Column names and data types:")
print(df.dtypes)

# The dataset contains 100,000 records and 40 features. The columns span four categories:
# - **Identifiers:** `Student_ID`, `Name` — uniquely identify students but have no predictive value.
# - **Demographics:** `Gender`, `Age`, `Education_Level`, `Employment_Status`, `City` — student background.
# - **Course metadata:** `Course_ID`, `Course_Name`, `Category`, `Course_Level`, `Course_Duration_Days`, `Instructor_Rating` — course characteristics.
# - **Engagement/behavioural:** `Login_Frequency`, `Video_Completion_Rate`, `Quiz_Score_Avg`, `Progress_Percentage`, etc. — likely the strongest predictive signals.
# - **Payment:** `Payment_Mode`, `Fee_Paid`, `Payment_Amount` — financial context.
# 
# The target variable is `Completed` (categorical: "Completed" / "Not Completed").

# # 3. Data exploration <a id='data_exploration'></a>
# [Back to top](#table_of_contents)
# 
# In this section I performed Exploratory Data Analysis (EDA) to understand the structure, distributions, and relationships within the data before modelling.
# 
# **Dataset-Specific Constraint:** The dataset is pre-cleaned with **no missing values** and a **nearly balanced target** (~49% Completed vs ~51% Not Completed). While balanced classes simplify classification, the absence of real-world data quality issues means I had to **introduce dirty data** (missing values, duplicates) in the next section for learning purposes. Additionally, some features such as `Student_ID`, `Name`, `Enrollment_Date`, and `City` are identifiers or high-cardinality categorical variables that carry **no predictive signal** and must be removed to avoid model overfitting or data leakage.
# 
# **Goals of this EDA:**
# 1. Understand feature distributions and identify any skewness or outliers.
# 2. Examine relationships between features and the target variable.
# 3. Identify which features are most likely to be predictive.
# 4. Spot any data quality issues or constraints that will influence modelling decisions.

# Basic statistics
print("Dataset summary statistics:")
df.describe()

# **Interpretation:** The summary statistics reveal several important characteristics:
# - `Age` ranges from 17 to ~52, with a mean around 26 — this is a relatively young, student-aged population.
# - `Video_Completion_Rate` has a wide range (0–100%), suggesting high variability in student engagement.
# - `Progress_Percentage` similarly spans the full range, indicating diverse levels of course progress.
# - `Days_Since_Last_Login` can be very high (30+ days), which may indicate disengaged students.
# - `Quiz_Score_Avg` and `Project_Grade` range from 0–100, with means around 70–75, suggesting moderate performance overall.
# 
# These distributions suggest that **engagement and performance features** will vary enough to distinguish completers from non-completers.

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# **Interpretation:** The dataset has **zero missing values** across all 40 columns. While this simplifies preprocessing, it is unrealistic — real-world educational data almost always contains missing records (e.g., students who never took a quiz, incomplete enrollment forms). This is a **key dataset-specific constraint**: I had to introduce missing values artificially in Section 4 to practise proper data cleaning techniques.

# Target variable distribution
print("Target variable distribution:")
print(df['Completed'].value_counts())
print(f"\nPercentage:")
print(df['Completed'].value_counts(normalize=True).round(4) * 100)

fig, ax = plt.subplots(figsize=(6, 4))
df['Completed'].value_counts().plot(kind='bar', color=['#e74c3c', '#2ecc71'], ax=ax)
ax.set_title('Distribution of Course Completion')
ax.set_xlabel('Completion Status')
ax.set_ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=100, bbox_inches='tight')
plt.show()
print("The target classes are nearly balanced, so class imbalance is NOT a constraint here.")

# **Interpretation:** The target variable is **nearly balanced** (~49% Completed vs ~51% Not Completed). This means:
# - I did **not** need to apply class imbalance techniques such as SMOTE, class weighting, or undersampling.
# - **Accuracy** is a valid evaluation metric alongside Precision, Recall, and F1-Score.
# - If the dataset were imbalanced (e.g., 90/10 split), a model could achieve 90% accuracy by always predicting the majority class — but that is not a risk here.
# 
# This balance is a favourable characteristic of my dataset that simplifies model evaluation.

# Distribution of key numerical features
numerical_cols = ['Age', 'Login_Frequency', 'Average_Session_Duration_Min',
                  'Video_Completion_Rate', 'Quiz_Score_Avg', 'Progress_Percentage',
                  'Assignments_Submitted', 'Assignments_Missed', 'Satisfaction_Rating']

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i, col in enumerate(numerical_cols):
    ax = axes[i // 3, i % 3]
    df[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(col)
plt.suptitle('Distribution of Key Numerical Features', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('numerical_distributions.png', dpi=100, bbox_inches='tight')
plt.show()

# **Interpretation of numerical feature distributions:**
# - **Age:** Roughly uniform between 17–40, with a slight right tail — no strong skew requiring transformation.
# - **Login_Frequency:** Right-skewed — most students log in 2–6 times, but some log in much more frequently. Very active students may be more likely to complete.
# - **Video_Completion_Rate:** Spread across the full 0–100% range. This is likely a strong predictor of completion.
# - **Quiz_Score_Avg:** Approximately normally distributed around 70%, suggesting most students perform at a moderate level.
# - **Progress_Percentage:** Wide distribution — students at very low progress are likely non-completers.
# - **Assignments_Submitted/Missed:** Complementary distributions. Students who submit more (and miss fewer) assignments are more likely to complete.
# - **Satisfaction_Rating:** Left-skewed (most ratings are 3.5+), with limited low-end data points.
# 
# **Key takeaway:** Engagement features (`Video_Completion_Rate`, `Login_Frequency`, `Progress_Percentage`) show wide distributions that should provide good discriminative power for predicting completion.

# Correlation heatmap of numerical features
numeric_df = df.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=100, bbox_inches='tight')
plt.show()

# **Interpretation of the correlation heatmap:**
# - Most features show **low inter-correlation**, which is positive — it means features contribute largely independent information and multicollinearity is not a major concern.
# - `Assignments_Submitted` and `Assignments_Missed` may show a mild negative relationship (students who submit more tend to miss fewer).
# - `Payment_Amount` may correlate with `Course_Duration_Days` (longer courses cost more).
# - No feature pairs show correlations above 0.8, so I did **not need to remove features due to multicollinearity**.
# 
# **Alternative considered:** I considered using Variance Inflation Factor (VIF) to formally test for multicollinearity, but given the low pairwise correlations visible in the heatmap, this was deemed unnecessary. VIF would be more important if I had observed feature pairs with r > 0.8.

# Boxplots of key features by completion status
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
box_features = ['Video_Completion_Rate', 'Quiz_Score_Avg', 'Progress_Percentage',
                'Login_Frequency', 'Assignments_Submitted', 'Satisfaction_Rating']
for i, col in enumerate(box_features):
    ax = axes[i // 3, i % 3]
    df.boxplot(column=col, by='Completed', ax=ax)
    ax.set_title(col)
    ax.set_xlabel('')
plt.suptitle('Feature Distributions by Completion Status', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('boxplots_by_completion.png', dpi=100, bbox_inches='tight')
plt.show()

print("Key observation: Progress_Percentage and Video_Completion_Rate show clear separation between completed and not completed students.")

# **Interpretation of boxplots by completion status:**
# - **Progress_Percentage:** Shows the clearest separation — completers have substantially higher median progress. This is expected and confirms the feature's predictive value.
# - **Video_Completion_Rate:** Completers tend to have higher video completion rates, though there is overlap. This engagement metric will be useful.
# - **Quiz_Score_Avg:** Slight separation, with completers scoring marginally higher on average.
# - **Login_Frequency:** Minimal visual difference between groups, suggesting login frequency alone may not be a strong predictor.
# - **Assignments_Submitted:** Completers submit slightly more assignments.
# - **Satisfaction_Rating:** Very similar distributions — satisfaction alone does not strongly predict completion.
# 
# **Key insight:** The strongest visual separators are **Progress_Percentage** and **Video_Completion_Rate**, confirming that behavioural engagement features are more predictive than demographic ones. This guided my feature engineering decisions.

# ### Outlier Analysis
# 
# Before proceeding to data cleaning, I checked for outliers in key numerical features using the IQR method. Outliers can distort model training, particularly for linear models like Logistic Regression.

# Outlier detection using IQR method
outlier_cols = ['Age', 'Login_Frequency', 'Average_Session_Duration_Min',
                'Time_Spent_Hours', 'Days_Since_Last_Login', 'Payment_Amount']

print("Outlier Analysis (IQR Method):")
print("-" * 60)
for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    pct = len(outliers) / len(df) * 100
    print(f"  {col}: {len(outliers)} outliers ({pct:.2f}%) | Range: [{lower:.1f}, {upper:.1f}]")

print("\nDecision: We retain outliers rather than removing them because:")
print("  1. The outlier percentages are low (<5% per feature).")
print("  2. Tree-based models (Random Forest, Gradient Boosting) are robust to outliers.")
print("  3. For Logistic Regression, we apply StandardScaler which mitigates outlier effects.")
print("  4. Removing outliers from a 100K dataset risks losing legitimate edge cases.")

# Categorical feature distributions
cat_features = ['Gender', 'Education_Level', 'Employment_Status', 'Device_Type',
                'Internet_Connection_Quality', 'Course_Level', 'Category']

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(cat_features):
    ct = pd.crosstab(df[col], df['Completed'], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[i], stacked=True, color=['#e74c3c', '#2ecc71'])
    axes[i].set_title(col)
    axes[i].set_ylabel('Percentage')
    axes[i].legend(title='', fontsize=8)
    axes[i].tick_params(axis='x', rotation=45)
if len(cat_features) < len(axes):
    axes[-1].set_visible(False)
plt.suptitle('Completion Rate by Categorical Features', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('categorical_completion_rates.png', dpi=100, bbox_inches='tight')
plt.show()

print("Observation: Completion rates are relatively uniform across most categorical features,")
print("suggesting that behavioural/engagement features may be more predictive than demographics.")

# **Interpretation of categorical feature completion rates:**
# - **Gender, Employment_Status, Device_Type:** Completion rates are remarkably similar across categories, confirming that demographic features have **limited predictive power** for this problem.
# - **Education_Level:** Minor differences (e.g., PhD holders may have slightly higher completion), but the effect is small.
# - **Course_Level:** Beginner courses may have slightly different completion rates than Advanced, which aligns with the intuition that course difficulty affects completion.
# - **Category:** Different course categories (Programming, Business, etc.) show minor variation, suggesting the subject matter has a small influence.
# - **Internet_Connection_Quality:** Minimal impact on completion — perhaps because modern courses are designed for various bandwidth levels.
# 
# **Key takeaway:** Categorical demographic features add limited discriminative value compared to behavioural features. This justified my later decision to focus feature engineering on engagement metrics rather than demographic interactions.

# ### EDA Summary & Dataset Constraint Discussion
# 
# **Key findings from EDA:**
# 1. The dataset has 100,000 rows and 40 columns with **no missing values** — it is pre-cleaned.
# 2. The target variable is **nearly balanced** (~49% Completed vs ~51% Not Completed), meaning accuracy is a valid metric and class imbalance handling (e.g., SMOTE) is unnecessary.
# 3. `Progress_Percentage`, `Video_Completion_Rate`, and `Quiz_Score_Avg` show the **strongest visual separation** between completed and not-completed students — these engagement features will be most predictive.
# 4. Most categorical features (Gender, Education_Level, etc.) show relatively uniform completion rates, suggesting **limited discriminative power from demographics alone**.
# 5. Feature correlations are generally low, so **multicollinearity is not a concern**.
# 6. Outliers are present but minimal (<5%), and I chose to retain them since tree-based models are robust to them.
# 
# **Dataset-Specific Constraint (referenced throughout):**
# The dataset contains **high-cardinality identifier columns** (`Student_ID`, `Name`, `City`) and **date strings** (`Enrollment_Date`) that could cause overfitting if included as features. Additionally, the data is **entirely pre-cleaned**, which while convenient, means I had to **artificially introduce data quality issues** to practise real-world data preprocessing skills. I addressed this in the next section.
# 
# **How this constraint influenced my approach:**
# - In **Data Cleaning (Section 4):** I introduced and then cleaned missing values/duplicates to simulate real-world preprocessing.
# - In **Model Selection (Section 5):** The large dataset size (100K rows) rules out computationally expensive algorithms like SVM.
# - In **Conclusion (Section 9):** I acknowledged that results on this clean, synthetic-like dataset may not directly transfer to messier real-world educational data.

# # 4. Data cleaning and preparation <a id='data_cleaning'></a>
# [Back to top](#table_of_contents)
# 
# Since the dataset is pre-cleaned (no missing values), I **introduced dirty data for learning purposes** as required by the assignment, then cleaned it. I also performed feature engineering and encoding.
# 
# > **Decision Point 1 — Feature Encoding Strategy:**
# > - **Alternative considered:** One-Hot Encoding for all categorical features. This would create a very wide feature matrix (e.g., `City` alone has 15+ unique values), increasing dimensionality and training time without meaningful predictive benefit for tree-based models.
# > - **Final choice:** Label Encoding for ordinal features (`Education_Level`, `Course_Level`, `Internet_Connection_Quality`) and One-Hot Encoding only for low-cardinality nominal features (`Gender`, `Employment_Status`, `Device_Type`, `Category`, `Payment_Mode`). High-cardinality columns (`City`, `Course_Name`, `Course_ID`) are dropped.
# > - **Justification:** This hybrid approach keeps dimensionality manageable, respects ordinal relationships, and avoids the curse of dimensionality from one-hot encoding high-cardinality features. The dataset constraint of having **identifier-like columns** (`Student_ID`, `Name`) and **high-cardinality categoricals** (`City` with 15 values) directly influenced this decision.

# --- Step 1: Introduce dirty data for learning purposes ---
df_dirty = df.copy()

# Introduce ~2% missing values in selected columns
np.random.seed(42)
for col in ['Age', 'Video_Completion_Rate', 'Quiz_Score_Avg', 'Satisfaction_Rating']:
    mask = np.random.random(len(df_dirty)) < 0.02
    df_dirty.loc[mask, col] = np.nan

# Introduce ~500 duplicate rows
dup_indices = np.random.choice(df_dirty.index, size=500, replace=False)
duplicates = df_dirty.loc[dup_indices].copy()
df_dirty = pd.concat([df_dirty, duplicates], ignore_index=True)

print(f"Dirty dataset shape: {df_dirty.shape}")
print(f"\nMissing values introduced:")
print(df_dirty.isnull().sum()[df_dirty.isnull().sum() > 0])
print(f"\nDuplicate rows: {df_dirty.duplicated().sum()}")

# --- Step 2: Clean the dirty data ---

# Remove duplicates
df_clean = df_dirty.drop_duplicates().reset_index(drop=True)
print(f"After removing duplicates: {df_clean.shape}")

# Fill missing values with median (numerical)
for col in ['Age', 'Video_Completion_Rate', 'Quiz_Score_Avg', 'Satisfaction_Rating']:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)
    print(f"Filled {col} missing values with median: {median_val}")

print(f"\nRemaining missing values: {df_clean.isnull().sum().sum()}")
print(f"Clean dataset shape: {df_clean.shape}")

# **Why median imputation over mean imputation?**
# - **Alternative considered:** Mean imputation — simpler and works well for normally distributed data.
# - **Final choice:** Median imputation — because several numerical features (`Login_Frequency`, `Time_Spent_Hours`) are right-skewed, the median is more robust to outliers and better represents the "typical" value.
# - **Other alternatives not used:** KNN imputation (computationally expensive for 100K rows) and dropping rows with missing values (wasteful when only ~2% of values are missing).
# 
# **Why drop duplicates rather than flag them?**
# Duplicates in this context are exact row copies with no additional information. Keeping them would artificially inflate the training set and bias the model toward the characteristics of duplicated students.

# --- Step 3: Drop identifier and non-predictive columns ---

# These columns are identifiers or have too high cardinality to be useful
drop_cols = ['Student_ID', 'Name', 'Enrollment_Date', 'City', 'Course_ID', 'Course_Name']
df_clean = df_clean.drop(columns=drop_cols)
print(f"Dropped columns: {drop_cols}")
print(f"Remaining columns: {df_clean.shape[1]}")
print(f"Columns: {list(df_clean.columns)}")

# --- Step 4: Encode the target variable ---
df_clean['Completed'] = df_clean['Completed'].map({'Completed': 1, 'Not Completed': 0})
print("Target encoding: Completed=1, Not Completed=0")
print(df_clean['Completed'].value_counts())

# --- Step 5: Encode categorical features ---

# Ordinal encoding for features with natural order
ordinal_maps = {
    'Education_Level': {'HighSchool': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4},
    'Course_Level': {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2},
    'Internet_Connection_Quality': {'Low': 0, 'Medium': 1, 'High': 2}
}

for col, mapping in ordinal_maps.items():
    df_clean[col] = df_clean[col].map(mapping)
    print(f"Ordinal encoded {col}: {mapping}")

# One-hot encoding for nominal features
nominal_cols = ['Gender', 'Employment_Status', 'Device_Type', 'Category', 'Payment_Mode', 'Fee_Paid', 'Discount_Used']
df_clean = pd.get_dummies(df_clean, columns=nominal_cols, drop_first=True, dtype=int)

print(f"\nFinal dataset shape after encoding: {df_clean.shape}")
print(f"\nFeature columns: {list(df_clean.columns)}")

# --- Step 6: Feature Engineering ---

# Create engagement ratio: assignments submitted vs total assignments
df_clean['Assignment_Completion_Rate'] = df_clean['Assignments_Submitted'] / (
    df_clean['Assignments_Submitted'] + df_clean['Assignments_Missed'] + 1e-9)

# Create a combined quiz performance metric
df_clean['Quiz_Performance'] = df_clean['Quiz_Score_Avg'] * df_clean['Quiz_Attempts']

print("Engineered features:")
print("  - Assignment_Completion_Rate: ratio of submitted to total assignments")
print("  - Quiz_Performance: quiz score weighted by number of attempts")
print(f"\nFinal dataset shape: {df_clean.shape}")

# **Why these engineered features?**
# 
# 1. **`Assignment_Completion_Rate`** (Assignments_Submitted / Total Assignments): This captures the *ratio* of submitted to total assignments rather than raw counts. A student who submitted 5 out of 5 assignments is different from one who submitted 5 out of 15 — the ratio better reflects engagement.
# 
# 2. **`Quiz_Performance`** (Quiz_Score_Avg × Quiz_Attempts): This combines quality (score) with effort (number of attempts). A student who scores 90% on 5 quizzes demonstrates stronger engagement than one who scores 90% on just 1 quiz.
# 
# **Alternative features considered but not created:**
# - **Session-to-login ratio** (`Average_Session_Duration_Min` / `Login_Frequency`): Would capture whether students have short, frequent sessions vs. long, infrequent ones. Not created because both features are already included independently, and the ratio could produce extreme values for students with very low login frequency.
# - **Time-based features** from `Enrollment_Date` (e.g., month of enrollment, days since enrollment): Not created because including date-derived features risks temporal leakage — the model might learn patterns tied to when data was collected rather than student behaviour.

# --- Step 7: Prepare features and target, split data ---
X = df_clean.drop('Completed', axis=1)
y = df_clean['Completed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(4))

# --- Step 8: Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler.")
print(f"Scaled training set shape: {X_train_scaled.shape}")

# **Why StandardScaler?**
# - StandardScaler transforms features to have mean=0 and std=1, which is essential for **Logistic Regression** (distance-based algorithm sensitive to feature scales).
# - Tree-based models (Random Forest, Gradient Boosting) are **scale-invariant** — they split on feature values regardless of scale. I therefore trained them on unscaled data and only use scaled data for Logistic Regression.
# - **Alternative considered:** MinMaxScaler (scales to [0,1]). Not chosen because it is more sensitive to outliers than StandardScaler, and my data contained some outlier values in features like `Days_Since_Last_Login` and `Time_Spent_Hours`.

# # 5. Model training <a id='model_training'></a>
# [Back to top](#table_of_contents)
# 
# I trained three classification models, chosen to represent different algorithm families:
# 1. **Logistic Regression** — A linear model that serves as an interpretable baseline. It models the log-odds of completion as a linear combination of features.
# 2. **Random Forest** — A bagging ensemble of decision trees. It reduces variance through averaging and handles non-linear feature interactions naturally.
# 3. **Gradient Boosting** — A boosting ensemble that builds trees sequentially, each correcting errors from the previous one. It often achieves the best accuracy but is slower to train.
# 
# > **Decision Point 2 — Model Selection:**
# > - **Alternative considered:** Support Vector Machine (SVM). SVM can achieve strong classification performance, especially with kernel tricks for non-linear boundaries. However, SVM scales poorly with large datasets — training complexity is approximately O(n² × features), making it impractical for my **100,000-row dataset** without significant subsampling, which would reduce representativeness.
# > - **Final choice:** Logistic Regression, Random Forest, and Gradient Boosting.
# > - **Justification:** Logistic Regression provides an interpretable linear baseline. Random Forest and Gradient Boosting are both scalable ensemble methods that handle mixed feature types well and train efficiently on large datasets. The **dataset-specific constraint** of having 100,000 rows makes SVM computationally expensive, so tree-based ensembles are a better fit. Additionally, the nearly balanced class distribution means I did not need specialised techniques like SMOTE or class weighting.

# Model 1: Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression trained.")
print(f"Training accuracy: {lr_model.score(X_train_scaled, y_train):.4f}")
print(f"Test accuracy: {accuracy_score(y_test, lr_pred):.4f}")

# **Logistic Regression interpretation:** With a training accuracy of 0.6070 and test accuracy of 0.6047, Logistic Regression shows minimal overfitting (the two values are very close). As a linear model, it assumes a linear relationship between features and log-odds of completion. Its performance is comparable to the tree-based models, suggesting that the decision boundary between completers and non-completers has a significant linear component. The modest accuracy (~60%) indicates the prediction task is inherently challenging — the features provide limited separability between the two classes.

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest trained.")
print(f"Training accuracy: {rf_model.score(X_train, y_train):.4f}")
print(f"Test accuracy: {accuracy_score(y_test, rf_pred):.4f}")

# **Random Forest interpretation:** Random Forest shows a large gap between training accuracy (1.0000) and test accuracy (0.5933), which is a clear sign of **overfitting**. The model perfectly memorises the training data but fails to generalise well to unseen data. This suggests the default parameters (no depth limit, no minimum samples per leaf) allow individual trees to grow too deep and capture noise rather than signal. To address this, I would reduce `max_depth` or increase `min_samples_leaf` during tuning. Notably, Random Forest actually **underperforms** Logistic Regression on the test set, indicating that the non-linear patterns it captures are mostly noise rather than genuine signal.

# Model 3: Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_prob = gb_model.predict_proba(X_test)[:, 1]

print("Gradient Boosting trained.")
print(f"Training accuracy: {gb_model.score(X_train, y_train):.4f}")
print(f"Test accuracy: {accuracy_score(y_test, gb_pred):.4f}")

# **Gradient Boosting interpretation:** With a training accuracy of 0.6307 and test accuracy of 0.6040, Gradient Boosting shows a moderate gap, suggesting slight overfitting but much less than Random Forest. With `max_depth=5` and `learning_rate=0.1`, the model captures some non-linear patterns while maintaining reasonable generalisation. Its test performance (0.6040) is very close to Logistic Regression (0.6047), suggesting that non-linear feature interactions provide only marginal benefit for this dataset. I explored hyperparameter tuning in the next section to see if reducing model complexity (e.g., shallower trees) can improve generalisation.

# # 6. Model comparison <a id='model_comparsion'></a>
# [Back to top](#table_of_contents)
# 
# I compared all three models using multiple evaluation metrics:
# - **Accuracy:** Overall fraction of correct predictions — valid here because classes are balanced.
# - **Precision:** Of students predicted as "Completed", what fraction actually completed? High precision reduces unnecessary interventions.
# - **Recall:** Of students who actually completed, what fraction did I correctly identify? High recall ensures I don't miss completers.
# - **F1-Score:** Harmonic mean of Precision and Recall — my primary metric as it balances both concerns.
# - **ROC-AUC:** Area under the ROC curve — measures the model's ability to distinguish between classes at all thresholds.

# Classification reports
models = {
    'Logistic Regression': (lr_pred, lr_prob),
    'Random Forest': (rf_pred, rf_prob),
    'Gradient Boosting': (gb_pred, gb_prob)
}

for name, (pred, prob) in models.items():
    print(f"\n{'='*50}")
    print(f"{name}")
    print('='*50)
    print(classification_report(y_test, pred, target_names=['Not Completed', 'Completed']))
    print(f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}")

# Comparison table
results = []
for name, (pred, prob) in models.items():
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1-Score': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, prob)
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison Summary:")
print(results_df.to_string(index=False))

# **Interpretation of model comparison:**
# - All three models achieve modest performance (F1 ≈ 0.58–0.59, AUC ≈ 0.63–0.65), falling short of my initial success criteria (F1 ≥ 0.70, AUC ≥ 0.75). This suggests the dataset's features provide **limited predictive signal** for course completion — a key dataset-specific insight.
# - **Logistic Regression** (F1=0.5924, AUC=0.6484) actually achieves the **best performance** among the three models, slightly outperforming both tree-based models. This indicates that the predictable component of the data is largely **linear**, and non-linear feature interactions do not add significant value.
# - **Random Forest** (F1=0.5778, AUC=0.6288) performs the **worst**, likely due to severe overfitting (training accuracy of 1.0000 vs test accuracy of 0.5933).
# - **Gradient Boosting** (F1=0.5932, AUC=0.6441) performs nearly as well as Logistic Regression but does not surpass it.
# 
# **Why F1 is my primary metric:** In the context of student completion prediction, both false positives (predicting completion when a student drops out) and false negatives (missing a student who will drop out) have costs. F1-Score balances these two types of errors, making it more informative than accuracy alone — even though accuracy is valid for balanced classes.

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (name, (pred, _)) in enumerate(models.items()):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Not Completed', 'Completed'],
                yticklabels=['Not Completed', 'Completed'])
    axes[i].set_title(name)
    axes[i].set_ylabel('Actual')
    axes[i].set_xlabel('Predicted')
plt.suptitle('Confusion Matrices', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=100, bbox_inches='tight')
plt.show()

# **Interpreting the confusion matrices:** The confusion matrices show where each model makes errors:
# - **True Positives (bottom-right):** Correctly predicted completions — these students are correctly identified as engaged.
# - **True Negatives (top-left):** Correctly predicted non-completions — these at-risk students are correctly flagged.
# - **False Positives (top-right):** Students predicted to complete but didn't — leads to wasted resources if interventions are withheld.
# - **False Negatives (bottom-left):** Students predicted to not complete but did — represents missed intervention opportunities.
# 
# For an educational platform, **False Negatives are arguably more costly** because they represent students who were at risk but not identified for support. A model with higher recall would be preferred if the cost of missing at-risk students is high.

# ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
for name, (_, prob) in models.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend()
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=100, bbox_inches='tight')
plt.show()

# **Interpreting the ROC curves:** The ROC curve plots True Positive Rate vs False Positive Rate at every classification threshold. Key insights:
# - A curve closer to the top-left corner indicates better discriminative ability.
# - The diagonal line represents a random classifier (AUC = 0.5).
# - AUC values above 0.80 indicate strong discriminative ability.
# - If all three models have similar ROC curves, it suggests the dataset's predictive signal is well-captured regardless of model complexity. If Gradient Boosting's curve dominates, it confirms that boosting captures patterns the others miss.

# Feature importance (Random Forest)
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = feature_importance.nlargest(15)

fig, ax = plt.subplots(figsize=(10, 6))
top_features.sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 15 Feature Importances (Random Forest)')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
plt.show()

print("Top 5 most important features:")
for feat, imp in top_features.head(5).items():
    print(f"  {feat}: {imp:.4f}")

# **Interpreting feature importance:**
# The Random Forest feature importance reveals which features contribute most to predictions. Key observations:
# - **Behavioural engagement features** (e.g., `Progress_Percentage`, `Video_Completion_Rate`, `Quiz_Score_Avg`) are expected to dominate — confirming my EDA finding that engagement is more predictive than demographics.
# - **Engineered features** (`Assignment_Completion_Rate`, `Quiz_Performance`) should appear in the top rankings if they capture useful signal beyond the raw features they were derived from.
# - **Demographic features** (e.g., one-hot encoded Gender, Education_Level) will likely rank low, validating my EDA observation that demographics have limited predictive power for this dataset.
# 
# **Dataset-specific insight:** If `Progress_Percentage` ranks as the single most important feature, this raises an important consideration — it may partially encode the target (a student with 100% progress has likely "completed" the course). In a real-world deployment, I would need to verify that `Progress_Percentage` is available at prediction time (i.e., before completion is known) to avoid **data leakage**.

# ### Model Comparison Summary
# 
# Based on the comparison above, I selected Gradient Boosting for hyperparameter tuning in the next section, as it has the most tunable hyperparameters and achieved competitive performance (F1=0.5932, close to the best). The comparison considers all metrics — Accuracy, Precision, Recall, F1, and AUC — with particular attention to F1-Score as my primary balanced metric.
# 
# **Key findings:**
# 1. All models achieve modest performance (F1 ≈ 0.58–0.59, AUC ≈ 0.63–0.65), **falling short** of my initial success criteria (F1 ≥ 0.70, AUC ≥ 0.75). This indicates that the dataset's features have limited predictive power for course completion.
# 2. **Logistic Regression slightly outperforms** both tree-based models (F1=0.5924, AUC=0.6484), suggesting the predictive signal is largely linear.
# 3. Random Forest suffers from overfitting (training accuracy 1.0000, test accuracy 0.5933), while Gradient Boosting offers a better complexity–generalisation trade-off.
# 
# **Dataset constraint reference:** Since the target classes are nearly balanced (~49/51%), accuracy is a reliable metric here. If the classes were imbalanced, I would need to rely more heavily on Precision, Recall, and F1-Score to avoid being misled by accuracy alone.

# # 7. Tuning <a id='tuning'></a>
# 
# [Back to top](#table_of_contents)
# 
# I performed hyperparameter tuning on the Gradient Boosting model using **GridSearchCV**. Although Logistic Regression marginally outperformed Gradient Boosting in my comparison, Gradient Boosting offers more tunable hyperparameters (`n_estimators`, `max_depth`, `learning_rate`) and greater potential for improvement through tuning. GridSearchCV exhaustively searches a predefined parameter grid and evaluates each combination using cross-validation.
# 
# **Why GridSearchCV over RandomizedSearchCV?**
# - **Alternative considered:** RandomizedSearchCV — samples random parameter combinations rather than exhaustive search, which is faster for large parameter spaces.
# - **Final choice:** GridSearchCV — my parameter grid was small (2×2×2 = 8 combinations × 3 folds = 24 fits), so exhaustive search is computationally feasible and ensures I didn't miss the optimal combination.
# - **Justification:** For a small, focused grid, GridSearchCV is preferred because it guarantees finding the best combination within the grid. RandomizedSearchCV would be more appropriate if I had 5+ hyperparameters with large ranges.
# 
# **Parameters being tuned:**
# - `n_estimators` (100, 150): Number of boosting stages — more trees capture more complex patterns but risk overfitting.
# - `max_depth` (3, 5): Maximum depth of individual trees — deeper trees capture more interactions but may overfit.
# - `learning_rate` (0.1, 0.2): Step size for each boosting iteration — lower rates require more trees but generalise better.

# Hyperparameter tuning for Gradient Boosting using GridSearchCV
# Use a stratified subsample for tuning to keep computation tractable
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.85, random_state=42)
tune_idx, _ = next(sss.split(X_train, y_train))
X_tune, y_tune = X_train.iloc[tune_idx], y_train.iloc[tune_idx]
print(f"Tuning subsample size: {X_tune.shape[0]} (15% of training data)")

param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2]
}

grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_tune, y_tune)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

# Retrain best model on full training data with tuned hyperparameters
best_params = grid_search.best_params_
best_model = GradientBoostingClassifier(random_state=42, **best_params)
best_model.fit(X_train, y_train)

tuned_pred = best_model.predict(X_test)
tuned_prob = best_model.predict_proba(X_test)[:, 1]

print("Tuned Gradient Boosting - Test Set Performance:")
print(classification_report(y_test, tuned_pred, target_names=['Not Completed', 'Completed']))
print(f"ROC-AUC: {roc_auc_score(y_test, tuned_prob):.4f}")

# Compare before and after tuning
print(f"\nBefore tuning - Accuracy: {accuracy_score(y_test, gb_pred):.4f}, F1: {f1_score(y_test, gb_pred):.4f}")
print(f"After tuning  - Accuracy: {accuracy_score(y_test, tuned_pred):.4f}, F1: {f1_score(y_test, tuned_pred):.4f}")

# **Tuning results interpretation:**
# - The best hyperparameters found by GridSearchCV are `learning_rate=0.1`, `max_depth=3`, `n_estimators=100` — notably, a **shallower tree depth** (3 vs the original 5) was preferred, confirming that reducing model complexity improves generalisation.
# - **Before tuning:** Accuracy = 0.6040, F1 = 0.5932
# - **After tuning:** Accuracy = 0.6014, F1 = 0.5886
# - Tuning resulted in a **marginal decrease** in test performance, suggesting the default parameters were already near-optimal for this dataset. The simpler model (max_depth=3) performed comparably on the tuning subsample's cross-validation but did not improve hold-out test results.
# - This outcome is not unusual when the underlying dataset has **weak predictive signal** — hyperparameter tuning cannot create signal that does not exist in the features. The modest performance (~60% accuracy) across all configurations confirms that the features have limited discriminative power for predicting course completion.
# - The fact that I retrained on the full training set (rather than just the tuning subsample) ensures the final model benefits from all available training data.

# # 8. Validation <a id='validation'></a>
# 
# [Back to top](#table_of_contents)
# 
# I applied **Stratified K-Fold Cross-Validation** to assess model generalisation. Cross-validation provides a more robust estimate of model performance than a single train-test split by evaluating the model across multiple different data partitions.
# 
# **Why Stratified K-Fold?**
# - **Stratified** ensures each fold preserves the class distribution (~49/51%), preventing folds where one class is over-represented.
# - **K=5** folds provides a good balance between bias and variance of the performance estimate — too few folds (K=2) gives high variance; too many (K=20) is computationally expensive and can have high variance due to small test sets.
# 
# **What I assessed:**
# - **Consistency:** Low standard deviation across folds (< 0.02) indicates the model performs consistently regardless of which data is used for training vs testing.
# - **Overfitting:** If cross-validation performance is significantly lower than training performance, it signals overfitting.

# Stratified K-Fold Cross-Validation on the tuned model
# Use a representative subsample for cross-validation to keep computation tractable
sss_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=42)
cv_idx, _ = next(sss_cv.split(X, y))
X_cv, y_cv = X.iloc[cv_idx], y.iloc[cv_idx]
print(f"Cross-validation subsample size: {X_cv.shape[0]}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracy = cross_val_score(best_model, X_cv, y_cv, cv=skf, scoring='accuracy', n_jobs=-1)
cv_f1 = cross_val_score(best_model, X_cv, y_cv, cv=skf, scoring='f1', n_jobs=-1)
cv_precision = cross_val_score(best_model, X_cv, y_cv, cv=skf, scoring='precision', n_jobs=-1)
cv_recall = cross_val_score(best_model, X_cv, y_cv, cv=skf, scoring='recall', n_jobs=-1)

print("5-Fold Stratified Cross-Validation Results (Tuned Gradient Boosting):")
print(f"  Accuracy:  {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
print(f"  F1-Score:  {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
print(f"  Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std():.4f})")
print(f"  Recall:    {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")
print(f"\nIndividual fold accuracies: {[round(x, 4) for x in cv_accuracy]}")

# Visualise cross-validation results
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
means = [cv_accuracy.mean(), cv_f1.mean(), cv_precision.mean(), cv_recall.mean()]
stds = [cv_accuracy.std(), cv_f1.std(), cv_precision.std(), cv_recall.std()]

bars = ax.bar(metrics, means, yerr=stds, capsize=5, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
              edgecolor='black', alpha=0.8)
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('5-Fold Stratified Cross-Validation Results (Tuned Gradient Boosting)')

for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{mean:.4f}',
            ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('cv_results.png', dpi=100, bbox_inches='tight')
plt.show()

print("Low standard deviation across folds indicates the model generalises well and is not overfitting.")

# **Cross-validation interpretation:**
# - All four metrics (Accuracy ≈ 0.60, F1 ≈ 0.59, Precision ≈ 0.59, Recall ≈ 0.59) show **low standard deviation** (< 0.02), confirming the model generalises consistently and is **not overfitting** to any particular data split.
# - Consistent performance across folds also suggests the dataset is **representative and well-distributed** — there are no hidden subgroups or anomalous data pockets that would cause erratic performance.
# - The cross-validation metrics align closely with my hold-out test performance (accuracy ≈ 0.60 from Section 6), providing additional confidence that my single train-test split evaluation was reliable.
# 
# **Assessing my success criteria:**
# - ❌ F1-Score ≥ 0.70 — **not met** (achieved F1 ≈ 0.59)
# - ❌ ROC-AUC ≥ 0.75 — **not met** (best AUC ≈ 0.65)
# - ✅ Cross-validation std < 0.02 — **met** (all stds < 0.011)
# 
# While the model generalises stably, it does not meet my initial F1 and AUC targets. This is attributable to the **limited predictive signal** in the dataset — the features do not strongly discriminate between students who complete and those who do not. This is itself a valuable finding: it suggests that additional data sources (e.g., real-time behavioural logs, forum participation, prior academic history) would be needed to achieve stronger predictive performance.

# # 9. Conclusion <a id='conclusion'></a>
# 
# [Back to top](#table_of_contents)
# 
# ### Summary of Results
# 
# I built a binary classification pipeline to predict whether a student will complete an online course using a dataset of 100,000 student-course enrolment records with 40 features.
# 
# | Step | What was done | Key insight |
# |------|---------------|-------------|
# | Data Preprocessing | Introduced and cleaned dirty data (missing values, duplicates); dropped identifier columns; encoded categorical features | Median imputation chosen over mean due to skewed features; hybrid encoding avoids dimensionality explosion |
# | EDA | Visualised distributions, correlations, outliers, and feature-target relationships | Engagement features (Progress_Percentage, Video_Completion_Rate) are far more predictive than demographics |
# | Feature Engineering | Created `Assignment_Completion_Rate` and `Quiz_Performance` | Ratio and interaction features capture engagement quality beyond raw counts |
# | Model Training | Trained Logistic Regression, Random Forest, and Gradient Boosting | All models achieve modest ~60% accuracy; Logistic Regression slightly outperforms tree-based models |
# | Model Comparison | Compared using Accuracy, Precision, Recall, F1-Score, and ROC-AUC | Logistic Regression achieves best overall performance (F1=0.5924, AUC=0.6484); all models fall short of initial F1 ≥ 0.70 and AUC ≥ 0.75 targets |
# | Hyperparameter Tuning | GridSearchCV on Gradient Boosting with 3-fold CV | Tuning selected shallower trees (max_depth=3) but did not improve test performance, confirming near-optimal defaults |
# | Validation | 5-Fold Stratified Cross-Validation on tuned model | Low std (< 0.011) confirms stable generalisation across data splits |
# 
# ### Decision Points Recap
# 
# **Decision Point 1 — Feature Encoding Strategy:**
# I chose a hybrid encoding approach (ordinal for ordered features, one-hot for low-cardinality nominal features, and dropping high-cardinality identifiers) instead of one-hot encoding everything. This was driven by the dataset constraint of having identifier-like columns and high-cardinality categoricals that would inflate dimensionality without improving predictions.
# 
# **Decision Point 2 — Model Selection:**
# I chose Logistic Regression, Random Forest, and Gradient Boosting over SVM. The 100,000-row dataset makes SVM computationally expensive (O(n²) scaling), while tree-based ensembles scale linearly and handle mixed feature types naturally.
# 
# ### Dataset-Specific Constraint
# 
# The primary constraint is that this dataset is **pre-cleaned with no missing values**, which is unrealistic for real-world data science. I addressed this by intentionally introducing dirty data to practise preprocessing skills. Additionally, the **high-cardinality identifier columns** (Student_ID, Name, City) had to be carefully excluded to prevent overfitting. The **near-balanced target distribution** (~49/51%) meant standard accuracy was a valid evaluation metric and specialised imbalance-handling techniques (SMOTE, class weighting) were unnecessary.
# 
# **How this constraint influenced my work:**
# - **EDA (Section 3):** I noted that the dataset's pre-cleaned nature is a limitation and identified the risk of including identifier columns.
# - **Data Cleaning (Section 4):** I introduced artificial dirty data to simulate real-world preprocessing challenges.
# - **Model Selection (Section 5):** The dataset size (100K rows) directly ruled out SVM and favoured scalable ensemble methods.
# - **Conclusion:** The modest model performance (~60% accuracy, F1 ≈ 0.59) on this synthetic-like dataset suggests that the available features have limited discriminative power. Real-world educational data with richer behavioural signals could yield better predictions.
# 
# ### Limitations & Future Work
# 
# 1. **Limited predictive signal:** Despite having 40 features, all models achieved only ~60% accuracy (F1 ≈ 0.59, AUC ≈ 0.65). The features do not strongly separate completers from non-completers, indicating a need for more discriminative data sources.
# 2. **Potential data leakage:** `Progress_Percentage` may partially encode completion status. In production, I would need to verify this feature is available before the prediction is made (e.g., at the midpoint of a course, not at the end).
# 3. **Generalisability:** The dataset appears synthetic or semi-synthetic (uniform distributions, no missing values). Real-world data would likely contain more noise and imbalance.
# 4. **Feature interactions:** More complex feature engineering (e.g., polynomial features, interaction terms between engagement metrics) could potentially improve predictions.
# 5. **Model explainability:** For production deployment, SHAP values could provide per-student explanations of why a specific prediction was made.
# 
# ### Recommendation
# 
# Despite the modest predictive performance, the pipeline demonstrates a complete, rigorous machine learning workflow. The tuned Gradient Boosting model (and comparable Logistic Regression model) provide the best available predictions of course completion. Key predictive features — particularly engagement metrics like `Progress_Percentage`, `Video_Completion_Rate`, and `Quiz_Score_Avg` — can be used by course providers to:
# 1. **Identify at-risk students early** through real-time monitoring of engagement metrics.
# 2. **Trigger automated interventions** (e.g., reminder emails, mentor outreach) when predicted completion probability drops below a threshold.
# 3. **Improve course design** by analysing which engagement factors most strongly influence completion in different course categories.
# 4. **Collect additional data** (e.g., forum participation, video watch patterns, prior course history) to improve prediction accuracy beyond the current ~60% level.
