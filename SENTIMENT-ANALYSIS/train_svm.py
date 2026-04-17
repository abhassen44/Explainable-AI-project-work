"""
train_svm.py — SVM Training Pipeline
======================================
Trains a Linear SVM with TF-IDF features for binary sentiment classification.

Pipeline:
  1. Load & preprocess text via preprocess.py (ML mode)
  2. TF-IDF vectorization (unigrams + bigrams)
  3. Train LinearSVC + CalibratedClassifierCV (for probability outputs)
  4. Evaluate on test set
  5. Save model + vectorizer artifacts

Usage:
  python train_svm.py
"""

import os
import pickle
import numpy as np
from preprocess import load_data, preprocess_for_ml, MODELS_DIR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score

# ─── Configuration ───────────────────────────────────────────
MAX_FEATURES = 50000
NGRAM_RANGE = (1, 2)
C_PARAM = 1.0
MAX_ITER = 10000


def main():
    print("=" * 60)
    print("  SVM Training Pipeline")
    print("=" * 60)

    # 1. Load data
    X_train_raw, X_test_raw, y_train, y_test = load_data()

    # 2. Preprocess for ML
    print("\n[INFO] Cleaning text...")
    X_train_clean = preprocess_for_ml(X_train_raw)
    X_test_clean = preprocess_for_ml(X_test_raw)

    # 3. TF-IDF Vectorization
    print("\n[INFO] Building TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    X_train_tfidf = tfidf.fit_transform(X_train_clean)
    X_test_tfidf = tfidf.transform(X_test_clean)

    print(f"  TF-IDF matrix shape: {X_train_tfidf.shape}")
    print(f"  Feature count: {len(tfidf.get_feature_names_out())}")

    # 4. Train LinearSVC
    print("\n[INFO] Training LinearSVC...")
    base_svm = LinearSVC(
        C=C_PARAM,
        max_iter=MAX_ITER,
        class_weight="balanced",
        random_state=42,
    )

    # Wrap in CalibratedClassifierCV for probability outputs (needed by LIME/SHAP)
    svm_model = CalibratedClassifierCV(base_svm, cv=5, method="sigmoid")
    svm_model.fit(X_train_tfidf, y_train)

    print("  Training complete.")

    # 5. Evaluate on test set
    print("\n[INFO] Evaluating on test set...")
    y_pred = svm_model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # 6. Save model and vectorizer
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(svm_model, f)
    print(f"[INFO] SVM model saved to {model_path}")

    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    print(f"[INFO] TF-IDF vectorizer saved to {tfidf_path}")

    print("\n" + "=" * 60)
    print("  SVM Training Complete!")
    print("=" * 60)

    return svm_model, tfidf


if __name__ == "__main__":
    main()
