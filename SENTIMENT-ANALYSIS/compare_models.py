"""
compare_models.py — Side-by-Side Model Comparison
===================================================
Generates comparison visualizations between CNN-LSTM and SVM:
  1. Grouped bar chart comparing all metrics
  2. Radar/spider chart for Accuracy, Precision, Recall, F1
  3. Comparison table (paper targets vs achieved)
  4. ROC curves overlay

Usage:
  python compare_models.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, roc_curve, auc,
)

matplotlib.use("Agg")

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "comparison")

# ─── Plot Style ──────────────────────────────────────────────
COLORS = {
    "cnn_lstm": "#6366f1",
    "svm": "#22c55e",
    "bg": "#0f172a",
    "card": "#1e293b",
    "text": "#e2e8f0",
    "grid": "#334155",
    "paper": "#f59e0b",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["card"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["text"],
    "text.color": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "grid.color": COLORS["grid"],
    "font.family": "sans-serif",
    "font.size": 12,
})

# Paper benchmark targets
PAPER_TARGETS = {
    "CNN-LSTM": {"Accuracy": 0.8729, "Precision": 0.8863, "Recall": 0.8556, "F1-Score": 0.8707},
    "SVM": {"Accuracy": 0.8721, "Precision": 0.8694, "Recall": 0.8758, "F1-Score": 0.8726},
}


def get_predictions():
    """Load both models and generate predictions."""
    from preprocess import load_data, preprocess_for_ml, preprocess_for_dl

    X_train_raw, X_test_raw, y_train, y_test = load_data()
    results = {}

    # SVM
    print("[INFO] Loading SVM model...")
    with open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb") as f:
        svm_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)

    X_test_ml = preprocess_for_ml(X_test_raw)
    X_test_tfidf = tfidf.transform(X_test_ml)
    svm_pred = svm_model.predict(X_test_tfidf)
    svm_prob = svm_model.predict_proba(X_test_tfidf)[:, 1]
    results["SVM"] = {"y_pred": svm_pred, "y_prob": svm_prob}

    # CNN-LSTM
    print("[INFO] Loading CNN-LSTM model...")
    from tensorflow.keras.models import load_model as keras_load
    cnn_model = keras_load(os.path.join(MODELS_DIR, "cnn_lstm_model.keras"))
    with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    _, X_test_pad, _ = preprocess_for_dl(X_train_raw, X_test_raw)
    cnn_prob = cnn_model.predict(X_test_pad, batch_size=64).flatten()
    cnn_pred = (cnn_prob > 0.5).astype(int)
    results["CNN-LSTM"] = {"y_pred": cnn_pred, "y_prob": cnn_prob}

    return y_test, results


def compute_metrics(y_true, y_pred):
    """Compute all metrics for a model."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
    }


def plot_grouped_bar(metrics_dict, save_path):
    """Plot 1: Grouped bar chart comparing all metrics."""
    metric_names = list(list(metrics_dict.values())[0].keys())
    models = list(metrics_dict.keys())
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model in enumerate(models):
        vals = [metrics_dict[model][m] for m in metric_names]
        color = COLORS["cnn_lstm"] if model == "CNN-LSTM" else COLORS["svm"]
        bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Metric", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("CNN-LSTM vs SVM — Performance Comparison", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  > Grouped bar chart saved to {save_path}")


def plot_radar(metrics_dict, save_path):
    """Plot 2: Radar/spider chart for core metrics."""
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor(COLORS["card"])

    for model_name, metrics in metrics_dict.items():
        values = [metrics[c] for c in categories]
        values += values[:1]
        color = COLORS["cnn_lstm"] if model_name == "CNN-LSTM" else COLORS["svm"]
        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Paper targets
    for model_name, targets in PAPER_TARGETS.items():
        values = [targets[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, "--", linewidth=1.5, alpha=0.5,
                label=f"{model_name} (Paper)", color=COLORS["paper"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Performance Radar — CNN-LSTM vs SVM", fontsize=16, fontweight="bold", pad=25)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  > Radar chart saved to {save_path}")


def plot_comparison_table(metrics_dict, save_path):
    """Plot 3: Comparison table — paper targets vs achieved."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    cols = ["Metric", "CNN-LSTM\n(Achieved)", "CNN-LSTM\n(Paper)", "SVM\n(Achieved)", "SVM\n(Paper)"]
    rows = []
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        row = [
            metric,
            f"{metrics_dict['CNN-LSTM'].get(metric, 0):.4f}",
            f"{PAPER_TARGETS['CNN-LSTM'].get(metric, 0):.4f}",
            f"{metrics_dict['SVM'].get(metric, 0):.4f}",
            f"{PAPER_TARGETS['SVM'].get(metric, 0):.4f}",
        ]
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center", colWidths=[0.15, 0.18, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS["cnn_lstm"] if col in [1, 2] else (COLORS["svm"] if col in [3, 4] else "#475569"))
            cell.set_text_props(fontweight="bold", color="white")
        else:
            cell.set_facecolor(COLORS["card"])
            cell.set_text_props(color=COLORS["text"])
        cell.set_edgecolor(COLORS["grid"])

    ax.set_title("Paper Targets vs Achieved Performance", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  > Comparison table saved to {save_path}")


def plot_roc_overlay(y_true, results, save_path):
    """Plot 4: Overlaid ROC curves for both models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, data in results.items():
        fpr, tpr, _ = roc_curve(y_true, data["y_prob"])
        roc_auc = auc(fpr, tpr)
        color = COLORS["cnn_lstm"] if model_name == "CNN-LSTM" else COLORS["svm"]
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{model_name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], color="#ef4444", lw=1.5, linestyle="--", alpha=0.5, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold")
    ax.set_title("ROC Curves — CNN-LSTM vs SVM", fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  > ROC overlay saved to {save_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Model Comparison — CNN-LSTM vs SVM")
    print("=" * 60)

    y_test, results = get_predictions()

    metrics_dict = {}
    for model_name, data in results.items():
        metrics_dict[model_name] = compute_metrics(y_test, data["y_pred"])
        print(f"\n  {model_name} Metrics:")
        for k, v in metrics_dict[model_name].items():
            print(f"    {k}: {v:.4f}")

    print("\n[INFO] Generating comparison plots...\n")

    plot_grouped_bar(metrics_dict, os.path.join(RESULTS_DIR, "grouped_bar_comparison.png"))
    plot_radar(metrics_dict, os.path.join(RESULTS_DIR, "radar_comparison.png"))
    plot_comparison_table(metrics_dict, os.path.join(RESULTS_DIR, "comparison_table.png"))
    plot_roc_overlay(y_test, results, os.path.join(RESULTS_DIR, "roc_overlay.png"))

    print(f"\n[INFO] All comparison plots saved to: {RESULTS_DIR}")
    print("=" * 60)
    print("  Comparison Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
