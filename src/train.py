"""
Model Training and Benchmarking Pipeline
=========================================
Trains and evaluates multiple models on the Credit Card Fraud Detection dataset:
1. Baseline Zero-Rule (Majority Class) Classifier
2. Logistic Regression From Scratch (Pure NumPy with class_weight='balanced')
3. Scikit-Learn Logistic Regression (class_weight='balanced')
4. Scikit-Learn Random Forest Classifier (class_weight='balanced')

Includes:
- Strict leakage-free preprocessing (StandardScaler fit only on train split)
- Validation-based decision threshold optimization (maximizing F1-score)
- Comprehensive metric evaluation (PR-AUC, ROC-AUC, Precision, Recall, F1)
- Generation and saving of publication-quality plots to outputs/
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.logistic_regression_scratch import LogisticRegressionScratch


def setup_plotting_style():
    """Configure modern plot aesthetics."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def load_and_preprocess_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load dataset and execute stratified train/test split with strict leakage prevention.
    """
    print(f"[1/6] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"      Total transactions: {len(df):,d} | Features: {df.shape[1] - 1}")

    X = df.drop(columns=["Class"])
    y = df["Class"].values

    # Stratified train/test split to preserve 0.17% fraud prevalence in both splits
    print("[2/6] Executing Stratified 80/20 Train-Test Split...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"      Training samples: {len(X_train_raw):,d} (Frauds: {int(y_train.sum())})")
    print(f"      Testing samples:  {len(X_test_raw):,d} (Frauds: {int(y_test.sum())})")

    # Fit scaler strictly on training split to prevent data leakage
    print("      Standardizing all features (StandardScaler fit strictly on train split)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, X.columns.tolist()


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray):
    """
    Calculate full evaluation metrics for binary classification.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc = roc_auc_score(y_true, y_proba)
    except Exception:
        roc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_proba)
    except Exception:
        pr_auc = 0.0

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "F1-Score": f1,
        "ROC-AUC": roc,
        "PR-AUC": pr_auc,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def find_optimal_threshold(model, X_train, y_train, random_state=42):
    """
    Tune decision threshold on a holdout validation set split from the training data.
    Maximizes F1-score without touching the test set.
    """
    print("[3/6] Tuning optimal decision threshold on holdout validation split...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # Predict probabilities on validation set
    if hasattr(model, "predict_proba"):
        val_proba = model.predict_proba(X_val)[:, 1]
    else:
        val_proba = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 91)
    f1_scores = []
    precisions = []
    recalls = []

    for t in thresholds:
        preds = (val_proba >= t).astype(int)
        f1_scores.append(f1_score(y_val, preds, zero_division=0))
        precisions.append(precision_score(y_val, preds, zero_division=0))
        recalls.append(recall_score(y_val, preds, zero_division=0))

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"      Optimal validation threshold: {best_threshold:.2f} (Val F1: {best_f1:.4f})")
    return best_threshold, thresholds, f1_scores, precisions, recalls


def main():
    setup_plotting_style()
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv"))
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    # 1. Load & Preprocess
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(data_path)

    # 2. Define Models
    print("\n[4/6] Initializing and training candidate models...")
    models = {}

    # Model 1: Baseline Majority Predictor
    print("  -> Training Baseline (Majority Classifier)...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    models["Majority Baseline"] = dummy

    # Model 2: Logistic Regression from Scratch
    print("  -> Training Custom Logistic Regression (Scratch, NumPy)...")
    start_time = time.time()
    scratch_lr = LogisticRegressionScratch(
        learning_rate=0.15,
        n_iterations=800,
        l2_penalty=0.005,
        class_weight="balanced",
        tolerance=1e-6,
        random_state=42,
        verbose=True,
    )
    scratch_lr.fit(X_train, y_train)
    scratch_time = time.time() - start_time
    print(f"     Custom Scratch Model finished in {scratch_time:.2f}s (Final Loss: {scratch_lr.loss_history[-1]:.4f})")
    models["Logistic Regression (Scratch)"] = scratch_lr

    # Model 3: Scikit-Learn Logistic Regression
    print("  -> Training Scikit-Learn Logistic Regression (Balanced)...")
    start_time = time.time()
    sk_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs")
    sk_lr.fit(X_train, y_train)
    sk_lr_time = time.time() - start_time
    print(f"     Sklearn Logistic Regression finished in {sk_lr_time:.2f}s")
    models["Scikit-Learn Logistic Regression"] = sk_lr

    # Model 4: Scikit-Learn Random Forest Classifier
    print("  -> Training Scikit-Learn Random Forest Classifier...")
    start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_time
    print(f"     Random Forest finished in {rf_time:.2f}s")
    models["Random Forest Classifier"] = rf

    # 3. Decision Threshold Tuning for Custom Scratch Model
    best_th, th_range, f1_vals, p_vals, r_vals = find_optimal_threshold(
        scratch_lr, X_train, y_train
    )

    # 4. Comprehensive Evaluation on Unseen Test Set
    print("\n[5/6] Evaluating all models on unseen test split (N=56,962 transactions)...")
    results = []
    predictions_dict = {}
    probabilities_dict = {}

    for name, model in models.items():
        if name == "Majority Baseline":
            preds = model.predict(X_test)
            proba = np.zeros(len(y_test))
        else:
            proba = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)

        probabilities_dict[name] = proba
        predictions_dict[name] = preds
        metrics = evaluate_model(name, y_test, preds, proba)
        results.append(metrics)

    # Add scratch model with optimal threshold
    scratch_proba = probabilities_dict["Logistic Regression (Scratch)"]
    scratch_tuned_preds = (scratch_proba >= best_th).astype(int)
    tuned_name = f"Logistic Regression (Scratch, Tuned Threshold={best_th:.2f})"
    probabilities_dict[tuned_name] = scratch_proba
    predictions_dict[tuned_name] = scratch_tuned_preds
    results.append(evaluate_model(tuned_name, y_test, scratch_tuned_preds, scratch_proba))

    # Format Results Table
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("MODEL BENCHMARK RESULTS (TEST SET EVALUATION)")
    print("=" * 90)
    cols_display = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "TP", "FP", "FN"]
    print(df_results[cols_display].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 90)

    # 5. Visualizations Generation
    print("\n[6/6] Generating and saving evaluation figures to outputs/...")

    # Plot 07: Scratch Loss Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scratch_lr.loss_history, color="#1f77b4", linewidth=2.0, label="BCE Loss + L2")
    ax.set_title("Convergence of Logistic Regression From Scratch (Batch Gradient Descent)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Weighted Binary Cross-Entropy Loss", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "07_scratch_loss_curve.png")
    fig.savefig(loss_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {loss_path}")

    # Plot 08: ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in ["Logistic Regression (Scratch)", "Scikit-Learn Logistic Regression", "Random Forest Classifier"]:
        fpr, tpr, _ = roc_curve(y_test, probabilities_dict[name])
        auc_score = roc_auc_score(y_test, probabilities_dict[name])
        ax.plot(fpr, tpr, linewidth=2.0, label=f"{name} (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.5000)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "08_roc_curves_comparison.png")
    fig.savefig(roc_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {roc_path}")

    # Plot 09: Precision-Recall Curves Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in ["Logistic Regression (Scratch)", "Scikit-Learn Logistic Regression", "Random Forest Classifier"]:
        prec, rec, _ = precision_recall_curve(y_test, probabilities_dict[name])
        ap = average_precision_score(y_test, probabilities_dict[name])
        ax.plot(rec, prec, linewidth=2.0, label=f"{name} (PR-AUC = {ap:.4f})")
    baseline_pr = y_test.sum() / len(y_test)
    ax.axhline(baseline_pr, color="red", linestyle="--", label=f"Baseline Prevalence ({baseline_pr:.4f})")
    ax.set_title("Precision-Recall Curves (Critical for Imbalanced Fraud Detection)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    pr_path = os.path.join(output_dir, "09_pr_curves_comparison.png")
    fig.savefig(pr_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {pr_path}")

    # Plot 10: Confusion Matrices Side-by-Side
    models_to_plot = [
        ("Majority Baseline", predictions_dict["Majority Baseline"]),
        ("Custom LR (Scratch, th=0.50)", predictions_dict["Logistic Regression (Scratch)"]),
        (f"Custom LR (Scratch, th={best_th:.2f})", predictions_dict[tuned_name]),
        ("Random Forest Classifier", predictions_dict["Random Forest Classifier"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (title, p) in enumerate(models_to_plot):
        cm = confusion_matrix(y_test, p)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["Legit (0)", "Fraud (1)"],
            yticklabels=["Legit (0)", "Fraud (1)"],
        )
        axes[idx].set_title(title, fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("True Label", fontsize=10)

    plt.suptitle("Confusion Matrix Comparison Across Models & Thresholds", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "10_confusion_matrices.png")
    fig.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {cm_path}")

    # Plot 11: Threshold Tuning Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(th_range, f1_vals, label="F1-Score", color="#2ca02c", linewidth=2.0)
    ax.plot(th_range, p_vals, label="Precision", color="#1f77b4", linestyle="--")
    ax.plot(th_range, r_vals, label="Recall", color="#d62728", linestyle="--")
    ax.axvline(best_th, color="black", linestyle=":", label=f"Optimal Threshold ({best_th:.2f})")
    ax.set_title("Decision Threshold Optimization on Validation Set", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Probability Decision Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    th_path = os.path.join(output_dir, "11_threshold_tuning_curve.png")
    fig.savefig(th_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {th_path}")

    print("\n[SUCCESS] Model training, evaluation, and visualization pipeline completed successfully!")


if __name__ == "__main__":
    main()
