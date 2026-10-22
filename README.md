# Credit Card Fraud Detection using Probability, Logistic Regression, and Optimization

A college-level Machine Learning & Data Science mini-project implementing binary classification for credit card fraud detection. The project explores the mathematics of Logistic Regression from scratch (using NumPy), optimization via gradient descent, class-weighted loss, threshold tuning, and rigorous performance evaluation on highly imbalanced data.

---

## 1. Project Directory Structure

```text
ML_MiniProject/
├── data/
│   └── creditcard.csv                 # Dataset (284,807 transactions; local only, gitignored)
├── notebooks/
│   └── fraud_detection.ipynb          # End-to-end interactive workflow, EDA, and model evaluation
├── src/
│   ├── logistic_regression_scratch.py # Pure NumPy Logistic Regression with balanced BCE loss
│   ├── train.py                       # Standalone model training, validation, and benchmarking
│   └── app.py                         # FastAPI backend serving live inference and frontend
├── frontend/                          # Modern glassmorphism web UI
│   ├── index.html                     # Semantic HTML5 layout, live predictor, and metrics
│   ├── index.css                      # Vanilla CSS with glowing dark mode aesthetics
│   └── app.js                         # Dynamic scoring, threshold tuning, and plot lightbox
├── models/
│   └── fraud_models.joblib            # Serialized model weights and sample test vectors
├── outputs/                           # High-resolution saved visualizations
│   ├── 01_class_imbalance.png
│   ├── 02_amount_distribution_by_class.png
│   ├── 03_time_distribution_by_class.png
│   ├── 04_feature_correlation_with_target.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_top_features_distribution.png
│   ├── 07_scratch_loss_curve.png
│   ├── 08_roc_curves_comparison.png
│   ├── 09_pr_curves_comparison.png
│   ├── 10_confusion_matrices.png
│   └── 11_threshold_tuning_curve.png
├── requirements.txt                   # Dependency specifications (NumPy, Scikit-Learn, FastAPI, etc.)
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

5. **Train Models & Regenerate Benchmark Plots:**
   ```bash
   python src/train.py
   ```

6. **Launch the FraudGuard AI Web Application:**
   ```bash
   python -m uvicorn src.app:app --reload --port 8000
   ```
   Open your browser at `http://127.0.0.1:8000` to interact with the live transaction scoring UI, threshold optimization sliders, and visual analytics gallery.

7. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/fraud_detection.ipynb
   ```

---

## 3. Project Roadmap

- [x] **Stage 1: Setup & Exploratory Data Analysis (EDA)**
  - Directory setup, environment verification, and dataset validation.
  - Descriptive statistics, missing values, duplicate investigation.
  - Class imbalance quantification and visualization (linear & log scale).
  - Transaction amount & temporal dynamic analysis.
  - Pearson correlation matrix and top predictive features identification.
- [x] **Stage 2: Data Preprocessing & Leakage Prevention**
  - Stratified 80:20 train-test split (`stratify=y`, `random_state=42`).
  - Feature scaling fitted strictly on training data (`StandardScaler`).
  - Strict isolation of test split to eliminate data leakage.
- [x] **Stage 3: Mathematical Foundations**
  - Linear combination, Sigmoid function, Log-loss / Binary Cross-Entropy derivation.
  - Analytical weight/bias gradients, L2 regularization, and balanced class-weighting.
- [x] **Stage 4: Logistic Regression from Scratch**
  - Pure NumPy implementation (`src/logistic_regression_scratch.py`).
  - Numerically stable sigmoid activation and clipped weighted BCE loss.
  - Batch Gradient Descent optimizer with convergence monitoring.
- [x] **Stage 5: Model Comparison & Benchmarking**
  - Majority Baseline vs Custom Scratch LR vs Scikit-Learn Balanced LR vs Random Forest.
- [x] **Stage 6: Comprehensive Metric Evaluation & Threshold Tuning**
  - PR-AUC, ROC-AUC, Precision, Recall, Specificity, F1-score.
  - Validation-only decision threshold optimization (improves F1 from 0.126 to 0.616).
- [ ] **Stage 7: Final Deliverables & Viva Preparation**
  - Results comparison tables, mini-project report outline, and viva Q&A.

---

## 4. Key EDA Findings (Measured on Real Data)

| Property / Check | Measured Result | Interpretation / Action |
| :--- | :--- | :--- |
| **Total Rows** | **284,807** | Substantial sample size for statistical analysis |
| **Total Columns** | **31** (30 features + 1 target) | Features `V1`–`V28` (PCA), `Time`, `Amount`, and `Class` |
| **Missing Values** | **0** (0.00%) | Complete dataset; no imputation required |
| **Duplicate Rows** | **1,081** | 1,062 legitimate, **19 fraud** (retained to prevent removing ~3.86% of rare frauds) |
| **Class 0 (Legitimate)** | **284,315** (99.8273%) | Majority class |
| **Class 1 (Fraudulent)**| **492** (0.1727%) | Minority class (1 fraud per ~578 legitimate transactions) |
| **Legitimate Amount** | Mean: \$88.29, Median: \$22.00, Max: \$25,691.16 | Highly right-skewed |
| **Fraudulent Amount** | Mean: \$122.21, Median: **\$9.25**, Max: \$2,125.87 | Fraudsters frequently test small transactions (< \$10) |
| **Top Negative Features** | `V17` (-0.326), `V14` (-0.303), `V12` (-0.261), `V10` (-0.217) | Lower values correlate with higher fraud probability |
| **Top Positive Features** | `V11` (+0.155), `V4` (+0.133), `V2` (+0.091) | Higher values correlate with higher fraud probability |

---

## 5. Model Benchmark Results (Evaluated on Unseen Test Split, N=56,962)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TP | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.9983 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0017 | 0 | 0 | 98 |
| **Logistic Regression (Scratch, $\\tau$=0.50)** | 0.9781 | 0.0676 | **0.9184** | 0.1259 | **0.9752** | 0.7138 | 90 | 1,242 | 8 |
| **Scikit-Learn Logistic Regression** | 0.9755 | 0.0610 | **0.9184** | 0.1144 | 0.9721 | 0.7190 | 90 | 1,386 | 8 |
| **Logistic Regression (Scratch, $\\tau$=0.95 Tuned)** | **0.9981** | **0.4775** | **0.8673** | **0.6159** | **0.9752** | 0.7138 | 85 | **93** | 13 |
| **Random Forest Classifier (Ensemble)** | **0.9994** | **0.8404** | **0.8061** | **0.8229** | 0.9732 | **0.8313** | 79 | 15 | 19 |

> **Key Takeaways:**
> 1. **Accuracy Paradox:** The Majority Baseline achieves 99.83% accuracy while completely missing all 98 fraud cases (F1 = 0.0).
> 2. **Custom NumPy Implementation Parity:** The Custom Scratch model trained with gradient descent equals/outperforms Scikit-Learn's L-BFGS solver in ROC-AUC (0.9752 vs 0.9721) and cuts false positives from 1,386 down to 1,242.
> 3. **Threshold Tuning Impact:** Optimizing the decision threshold on validation data ($\tau^* = 0.95$) reduces false positives from 1,242 to 93 (a 92.5% decrease) while retaining 86.73% recall, skyrocketing the F1-score from 0.1259 to 0.6159.

---

## 6. Visualizations Saved

Visualizations generated during the project are stored in `outputs/`:
- `01_class_imbalance.png`: Linear vs Log10 class count comparison.
- `02_amount_distribution_by_class.png`: Log-transformed amount boxplot and KDE.
- `03_time_distribution_by_class.png`: 48-hour transaction density comparison.
- `04_feature_correlation_with_target.png`: Pearson correlation bar chart ranked by predictive strength.
- `05_correlation_heatmap.png`: Full triangular feature correlation matrix.
- `06_top_features_distribution.png`: Boxplots of top 6 discriminative features.
- `07_scratch_loss_curve.png`: Convergence curve of Custom Scratch Gradient Descent.
- `08_roc_curves_comparison.png`: ROC Curves for Baseline, Scratch LR, Sklearn LR, and Random Forest.
- `09_pr_curves_comparison.png`: Precision-Recall Curves (the golden standard for imbalanced fraud).
- `10_confusion_matrices.png`: Confusion matrices side-by-side across all models and decision thresholds.
- `11_threshold_tuning_curve.png`: Validation curve showing Precision, Recall, and F1 tradeoff vs Threshold.
