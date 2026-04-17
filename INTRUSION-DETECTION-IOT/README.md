# 🛡️ IoT Intrusion Detection System with Explainable AI

A deep learning-based Intrusion Detection System (IDS) for IoT networks, built on the **NSL-KDD** dataset. The system maps attacks into 5 distinct categories (Normal, DoS, Probe, R2L, U2R) and classifies network traffic using three neural network architectures. It provides transparent decision explanations via **SHAP** and **LIME** through an interactive dashboard.

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

Traditional intrusion detection systems operate as black boxes — they flag threats but offer no reasoning. This project addresses that gap by combining deep learning classifiers with Explainable AI techniques:

1. **Detect** — Classify IoT network packets into normal traffic or specific attack categories (5-class target).
2. **Explain** — Use LIME and SHAP to show *which features* drove each specific prediction.
3. **Visualize** — Present real-time results and feature contributions through an interactive Streamlit dashboard.

---

## Architecture & Data Pipeline

### Models

| Model | Architecture | Purpose |
|-------|-------------|---------|
| **DNN** | 3 × Dense(64, ReLU) + Dropout(0.01) → Softmax (5) | Primary classifier (used in dashboard) |
| **1D-CNN** | 3 × [Conv1D → MaxPool → Dropout(0.3)] → Flatten | Sequential feature extraction |
| **2D-CNN** | 3 × [Conv2D → MaxPool → Dropout(0.01)] → Flatten | Spatial feature extraction (36 features reshaped to 6×6 image) |

### Preprocessing Pipeline

1. **Category Mapping:** specific attacks (neptune, satan, ipsweep) mapped to 5 umbrella categories.
2. **Label Encoding:** `protocol_type`, `service`, `flag`, and `label`.
3. **Feature Selection:** Pearson Correlation drops 5 highly-correlated features, keeping 36 core features.
4. **Normalization:** Min-Max Scaling (0–1 range).
5. **Training:** Model training utilizes Adam optimizer (lr=0.001) with `EarlyStopping` (patience=5).

---

## Evaluation & Comparison

The project includes custom scripts to generate comprehensive evaluation metrics for the academic paper:

- **`compare_models.py`**: Runs a side-by-side training and evaluation of the DNN and 2D-CNN. Outputs comparison PNGs of Accuracy, Loss, Confusion Matrices, ROC-AUC paths, and comprehensive metric tables (F1, MCC, Cohen's Kappa, etc.) in the `/comparison` folder.
- **`evaluate.py` & `evaluate_2dcnn.py`**: Generates standalone evaluation graphics for individual models (class distribution, metric tables).

---

```markdown
## Project Structure

```text
INTRUSION-DETECTION-IOT/
│
├── app.py                 # Streamlit dashboard (SHAP + LIME visualizations)
├── train_main.py          # Train DNN model & save preprocessors (5-class)
├── train_2dcnn.py         # Train 2D-CNN model (reshapes to 6x6)
├── compare_models.py      # Generates paper-accurate comparisons between DNN/2D-CNN
|
├── evaluate.py            # Generates evaluation metric PNGs for DNN
├── evaluate_2dcnn.py      # Generates evaluation metric PNGs for 2D-CNN
│
├── model_dnn.py           # DNN architecture builder
├── model_1dcnn.py         # 1D-CNN architecture builder
├── model_2dcnn.py         # 2D-CNN architecture builder
│
├── dnn_model.keras        # Saved trained DNN model for dashboard
├── 2dcnn_model.keras      # Saved trained 2D-CNN model
├── preprocessors.pkl      # Saved encoders, scaler, feature names for dashboard
│
├── KDDTrain+.txt          # NSL-KDD training dataset
└── KDDTest+.txt           # NSL-KDD testing dataset
```

---
```

## Setup & Installation

### 1. Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Models

To generate the dashboard requirements:
```bash
python train_main.py
```
*(Optional)* To train the 2D-CNN or run the model comparator:
```bash
python train_2dcnn.py
python compare_models.py
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

---

## Interactive Dashboard

The Streamlit dashboard (`app.py`) provides a hands-on method to test the trained DNN model.

1. **Select a sample** — Click "Generate Random Sample" to pull from background data, or manually adjust feature sliders.
2. **Inspect** — Click "Run Deep Inspection" to classify the traffic and view confidence scores.
3. **Explain** — Switch between LIME and SHAP tabs to see why the model made its decision.

---

## Explainable AI (XAI)

### LIME (Local Interpretable Model-agnostic Explanations)
- Generates a **local linear approximation** around the specific prediction.
- Shows exactly which feature values pushed the prediction toward or away from the specific class.
- Displayed as an embedded, interactive HTML table in the dashboard.

### SHAP (SHapley Additive exPlanations)
- Computes **Shapley values** using `shap.GradientExplainer`.
- Displays a horizontal bar chart of the top 10 most impactful features for the given prediction.
- 🔴 **Red bars** = feature pushes prediction *toward* the detected class.
- 🔵 **Cyan bars** = feature pushes prediction *away* from the detected class.
