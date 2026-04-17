"""
evaluate.py — Unified Evaluation Script
==========================================
Generates 8 publication-quality plots per model, plus a metrics summary.

Plots generated:
  1. confusion_matrix.png       — Heatmap with counts
  2. classification_report.png  — Precision/Recall/F1 horizontal bar chart
  3. roc_auc_curve.png          — ROC curve with AUC score
  4. accuracy_loss_summary.png  — Training history (CNN-LSTM) / CV scores (SVM)
  5. class_distribution.png     — Dataset positive/negative split
  6. wordcloud.png              — Word clouds for positive vs negative reviews
  7. precision_recall_curve.png — PR curve with AP score
  8. score_summary.png          — Table with all metrics

Usage:
  python evaluate.py --model svm
  python evaluate.py --model cnn_lstm
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    hamming_loss,
)
from wordcloud import WordCloud

matplotlib.use("Agg")

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─── Plot Style ──────────────────────────────────────────────
COLORS = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "positive": "#22c55e",
    "negative": "#ef4444",
    "bg": "#0f172a",
    "card": "#1e293b",
    "text": "#e2e8f0",
    "grid": "#334155",
}
plt.rcParams.update(
    {
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
    }
)


def load_model_and_predict(model_name):
    """Load model and generate predictions on test data."""
    from preprocess import load_data, preprocess_for_ml, preprocess_for_dl

    X_train_raw, X_test_raw, y_train, y_test = load_data()

    if model_name == "svm":
        # Load SVM artifacts
        with open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            tfidf = pickle.load(f)

        X_test_clean = preprocess_for_ml(X_test_raw)
        X_test_features = tfidf.transform(X_test_clean)

        y_pred = model.predict(X_test_features)
        y_pred_prob = model.predict_proba(X_test_features)[:, 1]

        return y_test, y_pred, y_pred_prob, X_test_raw, X_train_raw, y_train

    elif model_name == "cnn_lstm":
        from tensorflow.keras.models import load_model as keras_load

        model = keras_load(os.path.join(MODELS_DIR, "cnn_lstm_model.keras"))
        with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb") as f:
            tokenizer = pickle.load(f)

        _, X_test_pad, _ = preprocess_for_dl(X_train_raw, X_test_raw)

        y_pred_prob = model.predict(X_test_pad, batch_size=64).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

        return y_test, y_pred, y_pred_prob, X_test_raw, X_train_raw, y_train
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─── Plot Functions ──────────────────────────────────────────


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot 1: Confusion Matrix Heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdYlGn",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
        linewidths=0.5,
        linecolor=COLORS["grid"],
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  > Confusion matrix saved to {save_path}")


def plot_classification_report(y_true, y_pred, save_path):
    """Plot 2: Classification Report as horizontal bar chart."""
    report = classification_report(
        y_true, y_pred, target_names=["Negative", "Positive"], output_dict=True
    )

    classes = ["Negative", "Positive"]
    metrics = ["precision", "recall", "f1-score"]
    data = {m: [report[c][m] for c in classes] for m in metrics}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    width = 0.25

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["positive"]]
    for i, (metric, vals) in enumerate(data.items()):
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=colors[i])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xlabel("Class", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Classification Report", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  > Classification report saved to {save_path}")


def plot_roc_curve(y_true, y_pred_prob, save_path):
    """Plot 3: ROC-AUC Curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=COLORS["primary"], lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color=COLORS["negative"], lw=1.5, linestyle="--", alpha=0.7, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])

    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold")
    ax.set_title("ROC-AUC Curve", fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  > ROC-AUC curve saved to {save_path}")


def plot_training_history(model_name, save_path):
    """Plot 4: Training history (CNN-LSTM) or placeholder for SVM."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if model_name == "cnn_lstm":
        history_path = os.path.join(MODELS_DIR, "cnn_lstm_history.pkl")
        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                history = pickle.load(f)

            # Accuracy
            axes[0].plot(history["accuracy"], color=COLORS["primary"], lw=2, label="Train Accuracy")
            axes[0].plot(history["val_accuracy"], color=COLORS["positive"], lw=2, label="Val Accuracy")
            axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Accuracy")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # Loss
            axes[1].plot(history["loss"], color=COLORS["negative"], lw=2, label="Train Loss")
            axes[1].plot(history["val_loss"], color=COLORS["secondary"], lw=2, label="Val Loss")
            axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "No training history available", ha="center", va="center", fontsize=14)
    else:
        # SVM — show a summary card instead
        axes[0].text(
            0.5, 0.5,
            "SVM — No epoch-based\ntraining history\n\n(Trained with 5-fold\nCross-Validation)",
            ha="center", va="center", fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=1", facecolor=COLORS["primary"], alpha=0.3),
        )
        axes[0].set_title("Training Method", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        axes[1].text(
            0.5, 0.5,
            "LinearSVC + CalibratedClassifierCV\n\nC = 1.0\nmax_iter = 10000\nclass_weight = balanced",
            ha="center", va="center", fontsize=13,
            bbox=dict(boxstyle="round,pad=1", facecolor=COLORS["secondary"], alpha=0.3),
        )
        axes[1].set_title("Hyperparameters", fontsize=14, fontweight="bold")
        axes[1].axis("off")

    fig.suptitle(
        f"{'CNN-LSTM' if model_name == 'cnn_lstm' else 'SVM'} — Training Summary",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  > Training summary saved to {save_path}")


def plot_class_distribution(y_train, save_path):
    """Plot 5: Class distribution pie + bar chart."""
    counts = pd.Series(y_train).value_counts().sort_index()
    labels = ["Negative", "Positive"]
    colors_list = [COLORS["negative"], COLORS["positive"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    wedges, texts, autotexts = axes[0].pie(
        counts.values,
        labels=labels,
        colors=colors_list,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )
    axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold")

    # Bar chart
    bars = axes[1].bar(labels, counts.values, color=colors_list, edgecolor=COLORS["grid"], width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{val:,}",
            ha="center", fontsize=12, fontweight="bold",
        )
    axes[1].set_title("Sample Counts", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  > Class distribution saved to {save_path}")


def plot_wordclouds(X_raw, y_true, save_path):
    """Plot 6: Word clouds for positive and negative reviews."""
    from preprocess import clean_text

    positive_text = " ".join([clean_text(t) for t, y in zip(X_raw[:5000], y_true[:5000]) if y == 1])
    negative_text = " ".join([clean_text(t) for t, y in zip(X_raw[:5000], y_true[:5000]) if y == 0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    wc_pos = WordCloud(
        width=800, height=400,
        background_color=COLORS["bg"],
        colormap="Greens",
        max_words=100,
    ).generate(positive_text if positive_text else "empty")

    wc_neg = WordCloud(
        width=800, height=400,
        background_color=COLORS["bg"],
        colormap="Reds",
        max_words=100,
    ).generate(negative_text if negative_text else "empty")

    axes[0].imshow(wc_pos, interpolation="bilinear")
    axes[0].set_title("Positive Reviews", fontsize=14, fontweight="bold", color=COLORS["positive"])
    axes[0].axis("off")

    axes[1].imshow(wc_neg, interpolation="bilinear")
    axes[1].set_title("Negative Reviews", fontsize=14, fontweight="bold", color=COLORS["negative"])
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  > Word clouds saved to {save_path}")


def plot_precision_recall_curve(y_true, y_pred_prob, save_path):
    """Plot 7: Precision-Recall Curve with AP score."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color=COLORS["secondary"], lw=2.5, label=f"PR Curve (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.15, color=COLORS["secondary"])

    ax.set_xlabel("Recall", fontsize=13, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=13, fontweight="bold")
    ax.set_title("Precision-Recall Curve", fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  > Precision-Recall curve saved to {save_path}")


def plot_score_summary(y_true, y_pred, y_pred_prob, model_name, save_path):
    """Plot 8: Score summary table with all metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred),
        "Log Loss": log_loss(y_true, y_pred_prob),
        "Hamming Loss": hamming_loss(y_true, y_pred),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    table_data = [[k, f"{v:.4f}"] for k, v in metrics.items()]
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Score"],
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.8)

    # Style
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS["primary"])
            cell.set_text_props(fontweight="bold", color="white")
        else:
            cell.set_facecolor(COLORS["card"])
            cell.set_text_props(color=COLORS["text"])
        cell.set_edgecolor(COLORS["grid"])

    title = "CNN-LSTM" if model_name == "cnn_lstm" else "SVM"
    ax.set_title(f"{title} — Performance Metrics", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  > Score summary saved to {save_path}")


# ─── Main ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate a sentiment analysis model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["svm", "cnn_lstm"],
        help="Model to evaluate: 'svm' or 'cnn_lstm'",
    )
    args = parser.parse_args()

    model_name = args.model
    results_path = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(results_path, exist_ok=True)

    print("=" * 60)
    print(f"  Evaluating: {'CNN-LSTM' if model_name == 'cnn_lstm' else 'SVM'}")
    print("=" * 60)

    # Load model and get predictions
    y_test, y_pred, y_pred_prob, X_test_raw, X_train_raw, y_train = load_model_and_predict(model_name)

    # Generate all plots
    print("\n[INFO] Generating evaluation plots...\n")

    plot_confusion_matrix(y_test, y_pred, os.path.join(results_path, "confusion_matrix.png"))
    plot_classification_report(y_test, y_pred, os.path.join(results_path, "classification_report.png"))
    plot_roc_curve(y_test, y_pred_prob, os.path.join(results_path, "roc_auc_curve.png"))
    plot_training_history(model_name, os.path.join(results_path, "accuracy_loss_summary.png"))
    plot_class_distribution(y_train, os.path.join(results_path, "class_distribution.png"))
    plot_wordclouds(X_test_raw, y_test, os.path.join(results_path, "wordcloud.png"))
    plot_precision_recall_curve(y_test, y_pred_prob, os.path.join(results_path, "precision_recall_curve.png"))
    plot_score_summary(y_test, y_pred, y_pred_prob, model_name, os.path.join(results_path, "score_summary.png"))

    print(f"\n[INFO] All plots saved to: {results_path}")
    print("=" * 60)
    print("  Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
