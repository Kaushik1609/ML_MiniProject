# Credit Card Fraud Detection using Probability, Logistic Regression, and Optimization

A college-level Machine Learning & Data Science mini-project implementing binary classification for credit card fraud detection. The project explores the mathematics of Logistic Regression from scratch (using NumPy), optimization via gradient descent, class-weighted loss, threshold tuning, and rigorous performance evaluation on highly imbalanced data.

---

## 1. Project Directory Structure

```text
ML_MiniProject/
├── data/
│   └── creditcard.csv                 # Dataset (284,807 transactions; local only, gitignored)
├── notebooks/
│   └── fraud_detection.ipynb          # End-to-end interactive workflow and EDA
├── src/
│   └── logistic_regression_scratch.py # NumPy-only Logistic Regression (Stage 4)
├── outputs/                           # High-resolution saved visualizations
│   ├── 01_class_imbalance.png
│   ├── 02_amount_distribution_by_class.png
│   ├── 03_time_distribution_by_class.png
│   ├── 04_feature_correlation_with_target.png
│   ├── 05_correlation_heatmap.png
│   └── 06_top_features_distribution.png
├── requirements.txt                   # Dependency specifications
├── README.md                          # Project documentation and guide
└── .gitignore                         # Protects dataset and runtime artifacts from git
```

---

## 2. Environment Setup & Execution

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Git

### Installation

1. **Clone the repository / open project folder:**
   ```bash
   cd ML_MiniProject
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify dataset placement:**
   Ensure `creditcard.csv` is located at `data/creditcard.csv`.  
   *(If not present, place the Kaggle Credit Card Fraud Detection CSV into the `data/` folder).*

5. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/fraud_detection.ipynb
   ```

---

## 3. Project Roadmap

- [x] **Stage 1: Setup & Exploratory Data Analysis (EDA)** *(Current Stage)*
  - Directory setup, environment verification, and dataset validation.
  - Descriptive statistics, missing values, duplicate investigation.
  - Class imbalance quantification and visualization (linear & log scale).
  - Transaction amount & temporal dynamic analysis.
  - Pearson correlation matrix and top predictive features identification.
- [ ] **Stage 2: Data Preprocessing & Leakage Prevention**
  - Stratified 80:20 train-test split (`stratify=y`, `random_state=42`).
  - Feature scaling fitted strictly on training data (`RobustScaler` / `StandardScaler`).
  - Empirical evaluation of duplicate rows.
- [ ] **Stage 3: Mathematical Foundations**
  - Linear combination, Sigmoid function, Log-loss / Binary Cross-Entropy derivation.
  - Gradient descent, analytical weight/bias gradients, L2 regularization, class-weighting.
- [ ] **Stage 4: Logistic Regression from Scratch**
  - Pure NumPy implementation without Scikit-Learn `LogisticRegression`.
  - Numerically stable log-loss and clipping.
- [ ] **Stage 5: Model Comparison & Benchmarking**
  - Majority baseline vs. Custom Scratch Model vs. Scikit-learn Balanced vs. Random Forest.
- [ ] **Stage 6: Comprehensive Metric Evaluation**
  - PR-AUC, ROC-AUC, Precision, Recall, F1-score, and validation-only threshold tuning.
- [ ] **Stage 7: Final Deliverables & Viva Preparation**
  - Results comparison tables, mini-project report outline, and viva Q&A.

---

## 4. Key EDA Findings (Measured on Real Data)

| Property / Check | Measured Result | Interpretation / Action |
| :--- | :--- | :--- |
| **Total Rows** | **284,807** | Substantial sample size for statistical analysis |
| **Total Columns** | **31** (30 features + 1 target) | Features `V1`–`V28` (PCA), `Time`, `Amount`, and `Class` |
| **Missing Values** | **0** (0.00%) | Complete dataset; no imputation required |
| **Duplicate Rows** | **1,081** | 1,062 legitimate, **19 fraud** (dropping blindly removes ~3.86% of all fraud cases) |
| **Class 0 (Legitimate)** | **284,315** (99.8273%) | Majority class |
| **Class 1 (Fraudulent)**| **492** (0.1727%) | Minority class (1 fraud per ~578 legitimate transactions) |
| **Legitimate Amount** | Mean: \$88.29, Median: \$22.00, Max: \$25,691.16 | Highly right-skewed |
| **Fraudulent Amount** | Mean: \$122.21, Median: **\$9.25**, Max: \$2,125.87 | Fraudsters frequently test small transactions (< \$10) |
| **Top Negative Features** | `V17` (-0.326), `V14` (-0.303), `V12` (-0.261), `V10` (-0.217) | Lower values correlate with higher fraud probability |
| **Top Positive Features** | `V11` (+0.155), `V4` (+0.133), `V2` (+0.091) | Higher values correlate with higher fraud probability |

---

## 5. Visualizations Saved

Visualizations generated during Stage 1 are stored in `outputs/`:
- `01_class_imbalance.png`: Linear vs Log10 class count comparison.
- `02_amount_distribution_by_class.png`: Log-transformed amount boxplot and KDE.
- `03_time_distribution_by_class.png`: 48-hour transaction density comparison.
- `04_feature_correlation_with_target.png`: Pearson correlation bar chart ranked by predictive strength.
- `05_correlation_heatmap.png`: Full triangular feature correlation matrix.
- `06_top_features_distribution.png`: Boxplots of top 6 discriminative features.
