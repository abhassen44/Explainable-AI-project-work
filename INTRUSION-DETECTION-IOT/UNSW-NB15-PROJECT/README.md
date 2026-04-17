# 🛡️ UNSW-NB15 Intrusion Detection System with Explainable AI

A dedicated deep learning-based Intrusion Detection System (IDS) for IoT networks, built specifically on the **UNSW-NB15** dataset. The system filters and maps attacks strictly into 5 distinct target categories (Normal, Generic, Exploits, DoS, Fuzzers) and classifies network traffic using custom neural network architectures. It provides transparent decision explanations via **SHAP** and **LIME** through an interactive dashboard.

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

Traditional intrusion detection systems operate as black boxes — they flag threats but offer no reasoning. This subset project addresses that gap by applying deep learning classifiers mapped specifically to the UNSW-NB15 dataset:

1. **Detect** — Classify network packets into normal traffic or specific attack categories (5-class target).
2. **Explain** — Use LIME and SHAP to show *which features* drove each specific prediction.
3. **Visualize** — Present real-time results and feature contributions through an interactive Streamlit dashboard.

---

## Architecture & Data Pipeline

### Models

| Model | Architecture | Purpose |
|-------|-------------|---------|
| **DNN** | 3 × Dense(64, ReLU) → Softmax (5) | Primary classifier (used in dashboard) |
| **2D-CNN** | 3 × [Conv2D → MaxPool] → Flatten → Dense (5) | Spatial feature extraction (38 features padded to 49 and reshaped to 7×7 image) |

### Preprocessing Pipeline

1. **Category Filtering:** strictly isolates exactly 5 targeted classes (`Normal`, `Generic`, `Exploits`, `DoS`, `Fuzzers`).
2. **Feature Reduction:** Drops 6 irrelevant or highly-correlated features: `'ct_src_dport_ltm'`, `'loss'`, `'dwin'`, `'ct_ftp_cmd'`, `'label'`, and `'ct_srv_dst'` — resulting in 38 core features.
3. **Encoding:** Uses `LabelEncoder` for categorical structures (`proto`, `state`, `service`, `attack_cat`).
4. **Reshaping (2D-CNN only):** Pads the 38 features with 11 trailing zeros to perfectly fill a `7x7x1` input tensor.
5. **Normalization:** Uses Min-Max Scaling (0–1 range) across all inputs.
6. **Training:** Model training utilizes the Adam optimizer (lr=0.001) alongside weight decay (0.0001).

---

## Evaluation & Comparison

The project includes a unified script to evaluate both models and generate academic comparison plots:

- **`compare_models_unsw.py`**: Runs a side-by-side evaluation of the trained DNN and 2D-CNN models. Outputs comparison PNGs of Accuracy, Loss, Confusion Matrices, ROC-AUC paths, and comprehensive metric tables (F1, MCC, Cohen's Kappa, etc.) in the `/comparison` folder.

---

## Project Structure

```text
UNSW-NB15-PROJECT/
│
├── app_unsw.py                # Streamlit dashboard (SHAP + LIME visualizations)
├── train_unsw.py              # Main training script for both DNN and 2D-CNN
├── compare_models_unsw.py     # Evaluator script generating comparison charts
│
├── model_dnn_unsw.py          # DNN architecture builder (38 features)
├── model_2dcnn_unsw.py        # 2D-CNN architecture builder (7x7 features)
│
├── dnn_model_unsw.keras       # Saved trained DNN model (generated after training)
├── 2dcnn_model_unsw.keras     # Saved trained 2D-CNN model (generated after training)
├── preprocessors_unsw.pkl     # Saved encoders, scaler, background data (generated after training)
│
└── datasets/
    ├── UNSW_NB15_training-set.csv  # UNSW-NB15 training baseline
    └── UNSW_NB15_testing-set.csv   # UNSW-NB15 testing baseline
```

---

## Setup & Installation

Since this project resides inside the primary INTRUSION-DETECTION-IOT repository, it shares the exact same Python virtual environment.

### 1. Activate the Virtual Environment

```powershell
# From within the UNSW-NB15-PROJECT directory
..\.venv\Scripts\Activate.ps1
```

### 2. Train the Models

Run the complete pipeline to process the datasets and build your models.
```bash
python train_unsw.py
```

*(Optional)* Run the model comparator to dump comparison images to the `comparison/` folder:
```bash
python compare_models_unsw.py
```

### 3. Launch the Dashboard

```bash
streamlit run app_unsw.py
```

---

## Interactive Dashboard

The Streamlit dashboard (`app_unsw.py`) provides a hands-on method to test the trained Deep Neural Network.

1. **Select a sample** — Click "Generate Random Sample" to pull from background data, or manually adjust feature sliders.
2. **Inspect** — Click "Analyze Traffic Pattern" to classify the traffic and view confidence scores.
3. **Explain** — Switch between LIME and SHAP tabs to instantly see why the model made its decision.

---

## Explainable AI (XAI)

### LIME (Local Interpretable Model-agnostic Explanations)
- Generates a **local linear approximation** around the specific prediction.
- Shows exactly which feature values pushed the prediction toward or away from the specific class.
- Displayed as an embedded, interactive HTML table in the dashboard.

### SHAP (SHapley Additive exPlanations)
- Computes **Shapley values** using `shap.GradientExplainer`.
- Displays a horizontal bar chart of the top feature contributions for the given prediction.
- 🔴 **Red bars** = feature pushes prediction *toward* the detected class.
- 🔵 **Cyan bars** = feature pushes prediction *away* from the detected class.
