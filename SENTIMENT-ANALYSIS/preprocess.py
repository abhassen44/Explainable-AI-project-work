"""
preprocess.py — Shared Text Preprocessing Pipeline
====================================================
Provides text cleaning and two output modes:
  - preprocess_for_ml()  → cleaned strings for TF-IDF → SVM
  - preprocess_for_dl()  → tokenized + padded sequences for CNN-LSTM

Uses the pre-split train_data (1).csv and test_data (1).csv files.
"""

import re
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ─── Constants ───────────────────────────────────────────────
MAX_VOCAB_SIZE = 50000
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 128

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(DATA_DIR, "train_data (1).csv")
TEST_CSV = os.path.join(DATA_DIR, "test_data (1).csv")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# ─── Text Cleaning ──────────────────────────────────────────

# Initialize once
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def clean_text(text, remove_stopwords=True):
    """
    Core text cleaning pipeline:
    1. Remove HTML tags (<br />, <p>, etc.)
    2. Lowercase
    3. Remove special characters (keep alphanumeric + spaces)
    4. Tokenize
    5. Optionally remove stopwords
    6. Lemmatize
    7. Filter short tokens (len <= 2)
    """
    if not isinstance(text, str):
        return ""

    # HTML removal
    text = re.sub(r"<[^>]+>", " ", text)

    # Lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Stopword removal + Lemmatization
    if remove_stopwords:
        tokens = [
            _lemmatizer.lemmatize(t)
            for t in tokens
            if t not in _stop_words and len(t) > 2
        ]
    else:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens if len(t) > 2]

    return " ".join(tokens)


# ─── Data Loading ────────────────────────────────────────────


def load_data():
    """
    Load the pre-split train and test CSV files.
    The CSV has no header; columns are: text (col 0), sentiment (col 1).
    Returns: X_train, X_test, y_train, y_test
    """
    print("[INFO] Loading datasets...")

    train_df = pd.read_csv(TRAIN_CSV, header=None, names=["text", "sentiment"])
    test_df = pd.read_csv(TEST_CSV, header=None, names=["text", "sentiment"])

    # Drop any NaN rows
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples:  {len(test_df)}")
    print(
        f"  Train distribution: Positive={train_df['sentiment'].sum()}, "
        f"Negative={len(train_df) - train_df['sentiment'].sum()}"
    )

    X_train = train_df["text"].values
    X_test = test_df["text"].values
    y_train = train_df["sentiment"].values.astype(int)
    y_test = test_df["sentiment"].values.astype(int)

    return X_train, X_test, y_train, y_test


# ─── ML Preprocessing (for SVM) ─────────────────────────────


def preprocess_for_ml(texts, remove_stopwords=True):
    """
    Clean texts for ML pipeline (TF-IDF → SVM).
    Removes stopwords and lemmatizes.
    Returns: list of cleaned text strings.
    """
    print("[INFO] Preprocessing text for ML pipeline...")
    cleaned = [clean_text(t, remove_stopwords=remove_stopwords) for t in texts]
    print(f"  Cleaned {len(cleaned)} samples.")
    return cleaned


# ─── DL Preprocessing (for CNN-LSTM) ────────────────────────


def preprocess_for_dl(
    X_train, X_test, max_vocab=MAX_VOCAB_SIZE, max_len=MAX_SEQUENCE_LENGTH
):
    """
    Clean and tokenize texts for DL pipeline (CNN-LSTM).
    - Keeps stopwords (important for sequence context)
    - Uses Keras Tokenizer for integer encoding
    - Pads sequences to max_len

    Returns: X_train_padded, X_test_padded, tokenizer
    """
    print("[INFO] Preprocessing text for DL pipeline...")

    # Clean text (keep stopwords for DL — they carry sequential context)
    X_train_clean = [clean_text(t, remove_stopwords=False) for t in X_train]
    X_test_clean = [clean_text(t, remove_stopwords=False) for t in X_test]

    # Keras Tokenizer
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_clean)

    # Convert to integer sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train_clean)
    X_test_seq = tokenizer.texts_to_sequences(X_test_clean)

    # Pad sequences
    X_train_padded = pad_sequences(
        X_train_seq, maxlen=max_len, padding="post", truncating="post"
    )
    X_test_padded = pad_sequences(
        X_test_seq, maxlen=max_len, padding="post", truncating="post"
    )

    vocab_size = min(len(tokenizer.word_index) + 1, max_vocab)
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Train sequences: {X_train_padded.shape}")
    print(f"  Test sequences:  {X_test_padded.shape}")

    # Save tokenizer
    os.makedirs(MODELS_DIR, exist_ok=True)
    tokenizer_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"  Tokenizer saved to {tokenizer_path}")

    return X_train_padded, X_test_padded, tokenizer


# ─── Main (for testing) ─────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # Test ML preprocessing
    X_train_ml = preprocess_for_ml(X_train[:5])
    print("\n--- Sample ML Output ---")
    for i, txt in enumerate(X_train_ml):
        print(f"  [{i}] {txt[:100]}...")

    # Test DL preprocessing
    X_train_dl, X_test_dl, tok = preprocess_for_dl(X_train[:100], X_test[:100])
    print(f"\n--- Sample DL Output ---")
    print(f"  Shape: {X_train_dl.shape}")
    print(f"  Sample: {X_train_dl[0][:20]}")
