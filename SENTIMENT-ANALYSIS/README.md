# 🎭 Sentiment Analysis — Dual-Pathway with Explainable AI

A dual-pathway sentiment analysis system using **CNN-LSTM** and **SVM** models, with integrated **SHAP** and **LIME** explainability. Based on the research paper *"A Performance-Centric Evaluation of Machine Learning (ML) and Deep Learning (DL) Models for Sentiment and Opinion Analysis"* by Prabal Sharma & Anurag Rana.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture & Data Pipeline](#architecture--data-pipeline)
- [Evaluation & Comparison](#evaluation--comparison)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Interactive Dashboard](#interactive-dashboard)
- [Explainable AI (XAI)](#explainable-ai-xai)

---

## Overview

Traditional sentiment analysis systems operate as black boxes — they classify text but offer no reasoning. This project addresses that gap by combining machine learning and deep learning classifiers with Explainable AI techniques:

1. **Detect** — Classify movie reviews into Positive or Negative sentiment using a dual-pathway architecture.
2. **Explain** — Use LIME and SHAP to show *which words* drove each specific prediction.
3. **Visualize** — Present real-time results and word contributions through an interactive Streamlit dashboard.

---

## Architecture & Data Pipeline

### Models

| Model | Architecture | Purpose |
|-------|-------------|---------|
| **CNN-LSTM** | Embedding(128d) → Conv1D(128) → MaxPool(2) → LSTM(128) → Dense | Deep learning classifier, optimized for Precision (minimizing false positives) |
| **SVM** | TF-IDF(50K features, bigrams) → LinearSVC(C=1.0) | Machine learning classifier, optimized for Recall (maximizing positive coverage) |
| **Auto-Router** | Selects the best model based on priority flag | Dynamically routes input based on deployment priority (precision vs. recall) |

### Preprocessing Pipeline

1. **Cleaning:** Lowercasing, punctuation removal, and tag stripping.
2. **Stopwords:** Removed for SVM (TF-IDF), kept for CNN-LSTM to maintain sequence context.
3. **Vectorization (SVM):** TF-IDF vectorizer (up to 50K features, extracts bigrams).
4. **Tokenization (CNN-LSTM):** Keras Tokenizer mapping and padding/truncating sequences to max sequence length.
5. **Training:**
   - SVM utilizes `LinearSVC` with probability calibration (`CalibratedClassifierCV`).
   - CNN-LSTM utilizes `SpatialDropout(0.2)` and `Dropout(0.5)` to prevent overfitting, optimizing on binary crossentropy.

---

## Evaluation & Comparison

The project includes custom scripts to generate comprehensive evaluation metrics for the benchmark performance analysis:

- **`compare_models.py`**: Runs a side-by-side evaluation of CNN-LSTM and SVM models. Outputs comparison PNGs in the `/results/comparison` folder.
- **`evaluate.py`**: Generates standalone evaluation graphics for individual models (use `--model svm` or `--model cnn_lstm`). Outputs to the `/results/svm` or `/results/cnn_lstm` folders respectively.

**Performance Benchmarks**

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:---------|:----------|:-------|:---------|
| **CNN-LSTM** | 87.29% | 88.63% | 85.56% | 87.07% |
| **SVM** | 87.21% | 86.94% | 87.58% | 87.26% |

---

## Project Structure

```text
SENTIMENT-ANALYSIS/
│
├── app.py                 # Streamlit XAI dashboard (SHAP + LIME visualizations)
├── train_cnn_lstm.py      # Train CNN-LSTM model & save tokenizer
├── train_svm.py           # Train SVM model & save TF-IDF vectorizer
├── compare_models.py      # Generates comparisons between SVM/CNN-LSTM
|
├── evaluate.py            # Generates evaluation metric plots (8 per model)
├── preprocess.py          # Shared text preprocessing logic
│
├── model_cnn_lstm.py      # CNN-LSTM architecture builder
│
├── models/                # Saved model artifacts
│   ├── cnn_lstm_model.keras
│   ├── svm_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── tokenizer.pkl
│   └── preprocessors.pkl
│
├── datasets/              # IMDb dataset (auto-downloaded)
├── results/               # Evaluation & comparison plot outputs
│
├── setup_venv.ps1         # Environment setup script
├── requirements.txt       # Project dependencies
└── README.md              # This documentation file
```

---

## Setup & Installation

### 1. Create Virtual Environment

```powershell
cd SENTIMENT-ANALYSIS
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

### 2. Train the Models

```bash
python train_svm.py
python train_cnn_lstm.py
```

### 3. Evaluate

*(Optional)* To generate evaluation metrics and comparison plots:
```bash
python evaluate.py --model svm
python evaluate.py --model cnn_lstm
python compare_models.py
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

---

## Interactive Dashboard

The Streamlit dashboard (`app.py`) provides a hands-on method to test the trained dual-pathway system.

1. **Configure** — Select the model (CNN-LSTM, SVM, or Auto-Route) and the deployment priority (precision vs. recall).
2. **Select a sample** — Pick a predefined sample review from the dropdown or paste your own custom text.
3. **Inspect** — Click "Analyse" to classify the sentiment and view the confidence scores.
4. **Explain** — Switch between the LIME, SHAP, and Compare tabs to see exactly why the models made their decisions.

---

## Explainable AI (XAI)

### LIME (Local Interpretable Model-agnostic Explanations)
- Generates a **local linear approximation** around the specific prediction by perturbing words in the text.
- Shows exactly which words pushed the prediction toward positive or negative sentiment.
- Displayed as an embedded, interactive HTML text highlighter and feature weights chart in the dashboard.

### SHAP (SHapley Additive exPlanations)
- Computes **Shapley values** using coefficient analysis (SVM) or leave-one-out perturbation (CNN-LSTM).
- Displays a horizontal bar chart of the top 15 most impactful words for the given prediction.
- 🟢 **Green bars** = word pushes prediction *toward* positive sentiment.
- 🔴 **Red bars** = word pushes prediction *toward* negative sentiment.

---

## 📚 Reference

> Sharma, P., & Rana, A. (2024). *A Performance-Centric Evaluation of Machine Learning (ML) and Deep Learning (DL) Models for Sentiment and Opinion Analysis.*
