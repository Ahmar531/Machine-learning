# Adult Census Income — Feature Engineering

Feature engineering workflow for the **Adult Census Income** dataset. This project applies **16 feature engineering techniques** step by step — no model training, only data transformation and preparation.

**Dataset:** `c:\Users\user\OneDrive\Desktop\Dataset\adult.csv`  
**Notebook:** `adult_feature_engineering.ipynb`  
**Processed outputs:** `processed_data/` folder

---

## Dataset Overview

| Property | Value |
|----------|-------|
| Rows | 32,561 |
| Columns | 15 |
| Target | `income` (`<=50K` or `>50K`) |
| Missing values | Represented as `?` in workclass, occupation, native.country |

### Original Columns

| Column | Type | Description |
|--------|------|-------------|
| `age` | Numeric | Age in years |
| `workclass` | Categorical | Private, Self-emp, Gov, etc. |
| `fnlwgt` | Numeric | Final weight (sampling weight — dropped) |
| `education` | Categorical | Education level label (dropped, redundant) |
| `education.num` | Numeric | Education level (1–16, ordinal) |
| `marital.status` | Categorical | Marital status |
| `occupation` | Categorical | Job type |
| `relationship` | Categorical | Family relationship |
| `race` | Categorical | Race |
| `sex` | Categorical | Male / Female |
| `capital.gain` | Numeric | Capital gains |
| `capital.loss` | Numeric | Capital losses |
| `hours.per.week` | Numeric | Weekly working hours |
| `native.country` | Categorical | Country of origin |
| `income` | Categorical | Target: `<=50K` or `>50K` |

---

## Setup

```bash
pip install -r requirements.txt
```

### Run the Notebook

Open `adult_feature_engineering.ipynb` in Jupyter or VS Code and run all cells top to bottom.

---

## Feature Engineering Steps

### Step 1: Load & Explore Data

**Technique:** Exploratory Data Analysis (EDA)

Load the CSV with pandas and inspect:
- Shape, column names, data types
- Missing values (encoded as `?`)
- Target class distribution (imbalanced: ~76% `<=50K`, ~24% `>50K`)

**Why:** Understanding data quality and distribution before any transformation prevents errors downstream.

---

### Step 2: Handle Missing Values

**Technique:** Imputation

| Column Type | Strategy |
|-------------|----------|
| Categorical (`workclass`, `occupation`, `native.country`) | Mode (most frequent value) |
| Numerical | Median |

Replace `?` → `NaN`, then impute with `sklearn.impute.SimpleImputer`.

**Output:** `processed_data/step2_missing_handled.csv`

**Why:** Most ML algorithms cannot handle missing values. Mode/median imputation is simple and preserves distribution.

---

### Step 3: Drop Irrelevant Features

**Technique:** Feature Removal

| Dropped Column | Reason |
|----------------|--------|
| `fnlwgt` | Census sampling weight — not predictive of income |
| `education` | Redundant with `education.num` (numeric ordinal version) |

**Output:** `processed_data/step3_dropped_features.csv`

**Why:** Removing non-informative or redundant features reduces dimensionality and prevents multicollinearity.

---

### Step 4: Domain Feature Creation

**Technique:** Feature Engineering from Domain Knowledge

| New Feature | Formula / Logic |
|-------------|-----------------|
| `net_capital` | `capital.gain - capital.loss` |
| `has_capital_gain` | 1 if `capital.gain > 0`, else 0 |
| `has_capital_loss` | 1 if `capital.loss > 0`, else 0 |
| `work_intensity` | `hours.per.week / 40` |
| `is_full_time` | 1 if `hours.per.week >= 35` |

**Output:** `processed_data/step4_domain_features.csv`

**Why:** Combining raw columns into meaningful business features often improves model performance more than using raw values alone.

---

### Step 5: Binning / Discretization

**Technique:** Continuous → Categorical conversion

| Feature | Bins |
|---------|------|
| `age_group` | Young (0–25), Middle (25–40), Senior (40–55), Elderly (55+) |
| `hours_group` | Part-time (0–20), Full-time (20–40), Overtime (40–60), Heavy-overtime (60+) |

Uses `pandas.cut()` with predefined bin edges.

**Output:** `processed_data/step5_binned_features.csv`

**Why:** Binning captures non-linear relationships (e.g., income may plateau after a certain age) and makes patterns easier for tree-based models to split on.

---

### Step 6: Log Transformation

**Technique:** Logarithmic transformation (`np.log1p`)

Applied to highly right-skewed columns:
- `capital.gain`
- `capital.loss`
- `net_capital`

Creates: `capital.gain_log`, `capital.loss_log`, `net_capital_log`

**Output:** `processed_data/step6_log_transformed.csv`, `step6_log_transform.png`

**Why:** Capital gain/loss are heavily skewed (most values are 0, a few are very large). Log transform compresses the range and makes distributions closer to normal.

---

### Step 7: Outlier Treatment (IQR Capping)

**Technique:** Interquartile Range (IQR) capping

For each numeric column, values outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` are clipped to the boundary.

Applied to: `age`, `capital.gain`, `capital.loss`, `hours.per.week`, `education.num`

**Output:** `processed_data/step7_outlier_capped.csv`

**Why:** Extreme values can distort scaling and model training. Capping preserves data points while limiting their influence.

---

### Step 8: Label Encoding

**Technique:** Integer encoding for binary categories

| Column | Encoding |
|--------|----------|
| `sex` | Female=0, Male=1 |
| `income` (target) | `<=50K`=0, `>50K`=1 |

Uses `sklearn.preprocessing.LabelEncoder`.

**Output:** `processed_data/step8_label_encoded.csv`

**Why:** Converts categorical strings to numbers that algorithms can process. Best for binary or ordinal variables with clear ordering.

---

### Step 9: One-Hot Encoding

**Technique:** Dummy variable encoding

Converts nominal categories into binary (0/1) columns using `pd.get_dummies()` with `drop_first=True` to avoid the dummy variable trap.

Encoded columns:
- `workclass`, `marital.status`, `occupation`, `relationship`, `race`, `age_group`, `hours_group`

**Output:** `processed_data/step9_onehot_encoded.csv`

**Why:** Nominal categories have no natural order. One-hot encoding ensures the model treats each category independently without imposing false ordinal relationships.

---

### Step 10: Ordinal Encoding

**Technique:** Rank-based encoding

- `education.num` — already ordinal (1=lowest education, 16=doctorate)
- `country_rank_encoded` — rank countries by frequency in the dataset
- `country_freq_encoded` — map each country to its relative frequency

**Output:** `processed_data/step10_ordinal_encoded.csv`

**Why:** When categories have a natural order (education level) or when rank/frequency carries information (country), ordinal encoding preserves that structure in a single numeric column.

---

### Step 11: Frequency Encoding

**Technique:** Replace categories with their occurrence frequency

For high-cardinality columns (`native.country`, `occupation`, `workclass`):
```
category_frequency = count(category) / total_rows
```

**Output:** `processed_data/step11_frequency_encoded.csv`

**Why:** High-cardinality categoricals (40+ countries) create too many dummy columns with one-hot encoding. Frequency encoding compresses them into a single informative numeric feature.

---

### Step 12: Feature Scaling

**Technique:** Normalization / Standardization

Three scalers compared on numeric features:

| Scaler | Method | Best For |
|--------|--------|----------|
| `StandardScaler` | Mean=0, Std=1 | Normally distributed features |
| `MinMaxScaler` | Scale to [0, 1] | Bounded features, neural networks |
| `RobustScaler` | Median & IQR based | Data with outliers |

Train/test split: 70/30 with stratification on target.

**Output:** `processed_data/step12_train_scaled.csv`, `step12_test_scaled.csv`

**Why:** Features on different scales (age: 17–90, capital.gain: 0–99999) can cause distance-based and gradient-based algorithms to favor large-scale features.

---

### Step 13: Polynomial Features

**Technique:** Feature interaction & polynomial expansion

Uses `sklearn.preprocessing.PolynomialFeatures(degree=2)` on:
- `age`, `education.num`, `hours.per.week`

Generates original features + squared terms + pairwise interaction terms.

**Why:** Captures non-linear relationships and interactions (e.g., high education + many hours may jointly predict high income better than either alone).

---

### Step 14: Feature Selection

**Technique:** Statistical feature ranking

Two methods used:

| Method | Function | Purpose |
|--------|----------|---------|
| Mutual Information | `mutual_info_classif` | Measures dependency between feature and target |
| Chi-squared (χ²) | `SelectKBest(chi2, k=8)` | Selects top 8 non-negative features |

**Why:** Not all engineered features improve predictions. Feature selection removes noise and reduces overfitting.

---

### Step 15: Correlation Analysis

**Technique:** Pearson correlation matrix + heatmap

Visualizes pairwise correlations among all numeric features and the target variable.

**Output:** `processed_data/step15_correlation_matrix.png`

**Why:** Identifies multicollinearity (highly correlated features) and shows which features correlate most strongly with income.

---

### Step 16: Final Feature-Engineered Dataset

**Technique:** Assembly of best transformations

Combines the most useful engineered features into a single modeling-ready dataset:

```
age, education.num, capital.gain_log, capital.loss_log,
hours.per.week, net_capital_log, work_intensity, is_full_time,
has_capital_gain, has_capital_loss, sex_encoded,
country_freq_encoded, income_encoded
```

**Output:** `processed_data/final_feature_engineered.csv`

---

## Output Files Summary

| Step | File | Description |
|------|------|-------------|
| 2 | `step2_missing_handled.csv` | After imputation |
| 3 | `step3_dropped_features.csv` | After dropping fnlwgt & education |
| 4 | `step4_domain_features.csv` | With domain features |
| 5 | `step5_binned_features.csv` | With binned age & hours |
| 6 | `step6_log_transformed.csv` | With log-transformed columns |
| 6 | `step6_log_transform.png` | Before/after log transform plot |
| 7 | `step7_outlier_capped.csv` | With IQR-capped columns |
| 8 | `step8_label_encoded.csv` | With label-encoded sex & income |
| 9 | `step9_onehot_encoded.csv` | With one-hot dummy columns |
| 10 | `step10_ordinal_encoded.csv` | With ordinal country encoding |
| 11 | `step11_frequency_encoded.csv` | With frequency-encoded columns |
| 12 | `step12_train_scaled.csv` | Scaled training features |
| 12 | `step12_test_scaled.csv` | Scaled test features |
| 15 | `step15_correlation_matrix.png` | Correlation heatmap |
| 16 | `final_feature_engineered.csv` | Final modeling-ready dataset |

---

## Libraries Used

- **pandas** — data loading, manipulation, binning, one-hot encoding
- **numpy** — numerical operations, log transform
- **matplotlib / seaborn** — visualization, correlation heatmap
- **scikit-learn** — imputation, encoding, scaling, polynomial features, feature selection

---

## Project Structure

```
Feature Engineering/
├── adult_feature_engineering.ipynb   # Main notebook (16 steps)
├── feature.ipynb                     # Previous notebook (Social Network Ads)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── processed_data/                   # Output CSVs and plots (created on run)
    ├── step2_missing_handled.csv
    ├── step3_dropped_features.csv
    ├── ...
    └── final_feature_engineered.csv
```

---

## How To Run

1. Ensure the dataset exists at `c:\Users\user\OneDrive\Desktop\Dataset\adult.csv`
2. Install dependencies: `pip install -r requirements.txt`
3. Open `adult_feature_engineering.ipynb`
4. Run all cells sequentially
5. Check `processed_data/` for output files after each step

---

## Notes

- This project focuses **only on feature engineering** — no model training or evaluation.
- Each step saves an intermediate CSV so you can inspect transformations independently.
- The final dataset (`final_feature_engineered.csv`) is ready for any classification algorithm.
- Update `DATA_PATH` in the notebook if your CSV is in a different location.
