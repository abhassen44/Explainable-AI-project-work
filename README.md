#  Explainable AI (XAI) Project Work

This repository serves as a centralized portfolio for exploring and applying **Explainable Artificial Intelligence (XAI)** techniques. In modern machine learning, complex architectures like Deep Neural Networks or advanced ensemble models often act as "black boxes" — they provide high-accuracy predictions, but no reasoning behind those predictions. 

This project work aims to bridge the gap between high predictive performance and human interpretability by utilizing state-of-the-art XAI algorithms across three diverse computing domains.

---

##  What Are We Doing Through XAI?

The core objective across all sub-projects in this repository is to demonstrate *how* and *why* these black-box AI algorithms make their decisions. We achieve this transparency primarily through the integration of two leading XAI methodologies:

1. **SHAP (SHapley Additive exPlanations)** 
   - Based on cooperative game theory, SHAP computes the marginal contribution of every single input feature to the final prediction. It provides a robust, globally consistent way to see exactly which features drove a model's prediction higher or lower.
2. **LIME (Local Interpretable Model-agnostic Explanations)**
   - LIME approximates any black box model locally. By perturbing the input data slightly and seeing how the prediction changes, LIME creates a highly understandable, locally linear explanation of the model's behavior for individual samples.

By combining these visualization methodologies, our dashboards and analysis scripts elevate raw predictions into **trustworthy, actionable insights**. We focus on ensuring that the models are "right for the right reasons," completely eliminating blind trust in automated decisions.

---

## 📂 Project Sub-Domains

The repository is structured into three distinct projects, each applying XAI to a unique challenge:

### 🛡️ 1. Intrusion Detection in IoT Networks (`INTRUSION-DETECTION-IOT`)
- **Objective:** Classify network traffic into Normal traffic or specific attack categories (DoS, Probe, R2L, U2R).
- **Models Used:** Deep Neural Networks (DNN), 1D-CNN, 2D-CNN.
- **XAI Impact:** Network security analysts cannot automatically block traffic purely on a black-box suggestion. The dashboard uses LIME and SHAP to reveal exactly which packet features (e.g., source bytes, protocol type, failure flags) triggered an anomaly alert, allowing analysts to quickly verify and respond to real threats without false positives.

### 🎭 2. Sentiment Analysis & NLP (`SENTIMENT-ANALYSIS`)
- **Objective:** Evaluate movie reviews and categorize their implicit sentiment into Positive or Negative classes using a dual-pathway architecture.
- **Models Used:** Support Vector Machines (SVM) optimized for recall; CNN-LSTM optimized for precision.
- **XAI Impact:** We visualize natural language processing decisions by highlighting individual words. XAI calculates the weight of each word in the text sequence, indicating in real time whether linguistic markers pushed the model toward a positive or negative review. 

### ⚡ 3. Smart Electricity Framework (`SMART-ELECTRICITY-PROJECT`)
- **Objective:** Predict edge-based power consumption and perform anomaly analysis on smart home energy data.
- **Models Used:** Regression and Time-Series Analysis models based on the CHEFS Framework.
- **XAI Impact:** Identifying patterns in power consumption forecasting. Explainability here allows smart-grid operators and homeowners to understand which environmental or temporal factors most heavily correlate to spike predictions, justifying grid resource allocations.

---

##  Getting Started

Each project folder is self-contained with its own specific Virtual Environment configuration (`setup_venv.ps1`), `requirements.txt`, Model build scripts, evaluation tools, and interactive `app.py` or `dashboard.py` Streamlit applications.

Navigate to the respective child directories to view detailed installation instructions, architectural outlines, and research notes for each specific XAI domain!
