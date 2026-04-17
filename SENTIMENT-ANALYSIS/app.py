"""
app.py — Sentiment Analysis Dashboard with Explainable AI
==========================================================
Streamlit dashboard for movie review sentiment classification.

Models:
    - CNN-LSTM  (optimised for precision)
    - SVM       (optimised for recall)

XAI methods:
    - LIME  (Local Interpretable Model-agnostic Explanations)
    - SHAP  (SHapley Additive exPlanations, perturbation-based for DL)

Run:
    streamlit run app.py
"""

import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from lime.lime_text import LimeTextExplainer

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_STATS = {
    "precision": {
        "label":     "CNN-LSTM",
        "accuracy":  "87.29 %",
        "precision": "88.63 %",
        "recall":    "85.56 %",
        "f1":        "87.07 %",
        "blurb":     "Conv1D → LSTM → Dense. Minimises false positives.",
    },
    "recall": {
        "label":     "SVM",
        "accuracy":  "87.21 %",
        "precision": "86.94 %",
        "recall":    "87.58 %",
        "f1":        "87.26 %",
        "blurb":     "TF-IDF → LinearSVC. Maximises positive coverage.",
    },
}

SAMPLE_REVIEWS = {
    "Positive — action film":
        "This movie was absolutely incredible! The acting was superb, the special effects were "
        "breathtaking, and the story kept me on the edge of my seat the entire time. Highly recommended.",
    "Negative — weak plot":
        "I was really disappointed. The plot made no sense, the dialogue was terrible, and the "
        "characters were completely one-dimensional. One of the worst films I've seen this year.",
    "Positive — drama":
        "A beautifully crafted film. The performances are raw and authentic; the cinematography "
        "is stunning and the soundtrack perfectly complements every emotional beat.",
    "Negative — horror":
        "What a waste of time. Neither scary nor entertaining. Predictable jump scares, wooden acting, "
        "and a nonsensical ending. Save your money.",
    "Mixed review":
        "The lead actor gave a powerful performance, but the pacing was uneven and the third act "
        "fell apart completely. Great potential; disappointing execution.",
}


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analysis — XAI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* ── typography ── */
        .metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #111827;
        }

        /* ── metric tile ── */
        .metric-tile {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
        }

        /* ── prediction badge ── */
        .badge-pos {
            display: inline-block;
            background: #d1fae5;
            color: #065f46;
            border: 1px solid #6ee7b7;
            border-radius: 6px;
            padding: 0.35rem 0.9rem;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .badge-neg {
            display: inline-block;
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
            border-radius: 6px;
            padding: 0.35rem 0.9rem;
            font-weight: 600;
            font-size: 0.9rem;
        }

        /* ── result block ── */
        .result-block {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
        }
        .result-block p {
            margin: 0.25rem 0 0;
            font-size: 0.88rem;
            color: #6b7280;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_svm():
    with open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf


@st.cache_resource
def load_cnn_lstm():
    from tensorflow.keras.models import load_model

    model = load_model(os.path.join(MODELS_DIR, "cnn_lstm_model.keras"))
    with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "preprocessors.pkl"), "rb") as f:
        preprocessors = pickle.load(f)
    return model, tokenizer, preprocessors


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def clean_for_svm(text: str) -> str:
    from preprocess import clean_text
    return clean_text(text, remove_stopwords=True)


def clean_for_dl(text: str) -> str:
    from preprocess import clean_text
    return clean_text(text, remove_stopwords=False)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_svm(text: str, model, tfidf):
    cleaned  = clean_for_svm(text)
    features = tfidf.transform([cleaned])
    pred     = model.predict(features)[0]
    prob     = model.predict_proba(features)[0]
    return int(pred), prob, cleaned


def predict_cnn_lstm(text: str, model, tokenizer, preprocessors):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = clean_for_dl(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(
        seq,
        maxlen=preprocessors["max_sequence_length"],
        padding="post",
        truncating="post",
    )
    raw_prob = float(model.predict(padded, verbose=0)[0][0])
    pred     = 1 if raw_prob > 0.5 else 0
    prob     = np.array([1.0 - raw_prob, raw_prob])
    return pred, prob, cleaned


def route_prediction(text: str, priority: str):
    """
    Select model based on deployment priority.

    priority='precision'  →  CNN-LSTM  (precision 88.63 %)
    priority='recall'     →  SVM       (recall   87.58 %)
    """
    if priority == "precision":
        model, tokenizer, preprocessors = load_cnn_lstm()
        pred, prob, cleaned = predict_cnn_lstm(text, model, tokenizer, preprocessors)
        model_name = "CNN-LSTM"
    else:
        model, tfidf = load_svm()
        pred, prob, cleaned = predict_svm(text, model, tfidf)
        model_name = "SVM"

    return pred, prob, cleaned, model_name


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def generate_lime_explanation(text: str, model_name: str):
    explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

    if model_name == "SVM":
        model, tfidf = load_svm()

        def predict_fn(texts):
            cleaned  = [clean_for_svm(t) for t in texts]
            features = tfidf.transform(cleaned)
            return model.predict_proba(features)

    else:
        model, tokenizer, preprocessors = load_cnn_lstm()
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        def predict_fn(texts):
            cleaned = [clean_for_dl(t) for t in texts]
            seqs    = tokenizer.texts_to_sequences(cleaned)
            padded  = pad_sequences(
                seqs,
                maxlen=preprocessors["max_sequence_length"],
                padding="post",
                truncating="post",
            )
            probs = model.predict(padded, verbose=0).flatten()
            return np.column_stack([1.0 - probs, probs])

    return explainer.explain_instance(text, predict_fn, num_features=15, num_samples=500)


def render_lime_bar(explanation) -> plt.Figure:
    fig = explanation.as_pyplot_figure()

    fig.patch.set_facecolor("#ffffff")
    ax = fig.gca()
    ax.set_facecolor("#f9fafb")
    ax.set_title("LIME — feature weights", fontsize=13, fontweight="normal", color="#111827", pad=12)
    ax.set_xlabel("Weight", fontsize=11, color="#374151")
    ax.tick_params(colors="#374151")

    for spine in ax.spines.values():
        spine.set_edgecolor("#e5e7eb")

    for bar in ax.patches:
        bar.set_color("#059669" if bar.get_width() >= 0 else "#dc2626")
        bar.set_edgecolor("none")
        bar.set_alpha(0.85)

    fig.tight_layout()
    return fig


def render_lime_highlighted(explanation, raw_text: str) -> str:
    weights = dict(explanation.as_list())
    if not weights:
        return "<p style='color:#6b7280;font-size:0.9rem;'>No significant features found.</p>"

    max_w  = max(abs(v) for v in weights.values()) or 1.0
    tokens = raw_text.split()
    spans  = []

    for token in tokens:
        key = token.strip(".,!?;:'\"()[]{}").lower()
        if key in weights:
            w     = weights[key]
            alpha = min(abs(w) / max_w, 1.0) * 0.55 + 0.15
            color = (
                f"rgba(5,150,105,{alpha:.2f})"   if w > 0
                else f"rgba(220,38,38,{alpha:.2f})"
            )
            spans.append(
                f'<span style="background:{color};padding:2px 5px;border-radius:4px;'
                f'margin:1px;display:inline-block;">{token}</span>'
            )
        else:
            spans.append(f'<span style="margin:1px;display:inline-block;">{token}</span>')

    legend = (
        '<div style="margin-bottom:10px;font-size:0.82rem;color:#6b7280;">'
        '<span style="background:rgba(5,150,105,0.4);padding:2px 8px;border-radius:4px;'
        'margin-right:8px;">positive influence</span>'
        '<span style="background:rgba(220,38,38,0.4);padding:2px 8px;border-radius:4px;">'
        'negative influence</span>'
        '</div>'
    )
    body = " ".join(spans)
    return (
        f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;'
        f'padding:1rem 1.2rem;color:#111827;font-size:0.95rem;line-height:1.9;">'
        f'{legend}{body}</div>'
    )


# ---------------------------------------------------------------------------
# SHAP (perturbation-based for CNN-LSTM; coefficient-based for SVM)
# ---------------------------------------------------------------------------

def render_shap_bar(text: str, model_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#f9fafb")
    fig.patch.set_facecolor("#ffffff")

    if model_name == "SVM":
        model, tfidf = load_svm()
        cleaned      = clean_for_svm(text)
        features     = tfidf.transform([cleaned])

        feature_names   = tfidf.get_feature_names_out()
        nonzero_idx     = features.nonzero()[1]
        nonzero_vals    = features.toarray()[0, nonzero_idx]

        try:
            base_model   = model.calibrated_classifiers_[0].estimator
            coefficients = base_model.coef_[0]
            contributions = nonzero_vals * coefficients[nonzero_idx]
        except Exception:
            contributions = nonzero_vals

        word_contribs = sorted(
            zip(feature_names[nonzero_idx], contributions),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:15]

    else:
        model, tokenizer, preprocessors = load_cnn_lstm()
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        cleaned    = clean_for_dl(text)
        words      = cleaned.split()
        seq        = tokenizer.texts_to_sequences([cleaned])
        padded     = pad_sequences(
            seq,
            maxlen=preprocessors["max_sequence_length"],
            padding="post",
            truncating="post",
        )
        base_prob  = float(model.predict(padded, verbose=0)[0][0])

        raw_contribs = []
        for i, word in enumerate(words[:30]):
            perturbed      = words[:i] + words[i + 1:]
            p_seq          = tokenizer.texts_to_sequences([" ".join(perturbed)])
            p_padded       = pad_sequences(
                p_seq,
                maxlen=preprocessors["max_sequence_length"],
                padding="post",
                truncating="post",
            )
            new_prob       = float(model.predict(p_padded, verbose=0)[0][0])
            raw_contribs.append((word, base_prob - new_prob))

        word_contribs = sorted(raw_contribs, key=lambda x: abs(x[1]), reverse=True)[:15]

    if not word_contribs:
        ax.text(0.5, 0.5, "No significant features found.",
                ha="center", va="center", fontsize=13, color="#6b7280")
        return fig

    words_plot, values = zip(*word_contribs)
    colors = ["#059669" if v > 0 else "#dc2626" for v in values]

    y_pos = np.arange(len(words_plot))
    ax.barh(y_pos, values, color=colors, height=0.55, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words_plot, fontsize=11, color="#374151")
    ax.invert_yaxis()
    ax.axvline(x=0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Contribution towards positive sentiment", fontsize=11, color="#374151")
    ax.set_title(
        f"SHAP-style word contributions — {model_name}",
        fontsize=13, fontweight="normal", color="#111827", pad=12,
    )
    ax.tick_params(colors="#374151")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e5e7eb")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#059669", alpha=0.85, label="pushes → positive"),
            Patch(facecolor="#dc2626", alpha=0.85, label="pushes → negative"),
        ],
        loc="lower right",
        fontsize=10,
        framealpha=0.8,
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def build_sidebar() -> str:
    """Render sidebar controls; return the active priority ('precision'/'recall')."""
    with st.sidebar:
        st.markdown("## Configuration")
        st.divider()

        model_choice = st.radio(
            "Model",
            options=["CNN-LSTM (precision)", "SVM (recall)", "Auto-route"],
            index=2,
        )

        if model_choice == "Auto-route":
            priority = st.select_slider(
                "Priority",
                options=["recall", "precision"],
                value="precision",
                help="Precision → CNN-LSTM minimises false positives.\n"
                     "Recall → SVM maximises positive coverage.",
            )
        elif "CNN-LSTM" in model_choice:
            priority = "precision"
        else:
            priority = "recall"

        st.divider()

        info = MODEL_STATS[priority]
        st.markdown("**Active model**")
        st.markdown(
            f"**{info['label']}**  \n"
            f"{info['blurb']}  \n\n"
            f"Accuracy: {info['accuracy']}  \n"
            f"Precision: {info['precision']}  \n"
            f"Recall: {info['recall']}  \n"
            f"F1: {info['f1']}"
        )

        st.divider()
        st.caption(
            "Reference: *A Performance-Centric Evaluation of ML and DL Models "
            "for Sentiment Analysis* — Prabal Sharma & Anurag Rana"
        )

    return priority


# ---------------------------------------------------------------------------
# Metric strip
# ---------------------------------------------------------------------------

def render_metrics(priority: str) -> None:
    info = MODEL_STATS[priority]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [info["accuracy"], info["precision"], info["recall"], info["f1"]]

    cols = st.columns(4)
    for col, label, value in zip(cols, labels, values):
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

def render_input() -> str:
    col_text, col_samples = st.columns([3, 2])

    with col_text:
        st.markdown("#### Review text")
        user_text = st.text_area(
            label="review",
            label_visibility="collapsed",
            height=165,
            placeholder="Paste or type a movie review…",
            key="review_input",
        )

    with col_samples:
        st.markdown("#### Sample reviews")
        choice = st.selectbox(
            label="samples",
            label_visibility="collapsed",
            options=["— custom input —"] + list(SAMPLE_REVIEWS.keys()),
            key="sample_select",
        )
        if choice != "— custom input —":
            user_text = SAMPLE_REVIEWS[choice]
            st.text_area(
                label="preview",
                label_visibility="collapsed",
                value=user_text,
                height=125,
                disabled=True,
                key="sample_preview",
            )

    return user_text


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------

def render_prediction_result(pred: int, prob, model_used: str) -> None:
    label      = "Positive" if pred == 1 else "Negative"
    badge_cls  = "badge-pos" if pred == 1 else "badge-neg"
    confidence = max(prob) * 100

    st.markdown(
        f'<div class="result-block">'
        f'<span class="{badge_cls}">{label}</span>'
        f'<p>Confidence {confidence:.1f} % &nbsp;·&nbsp; Model: {model_used}'
        f'&nbsp;·&nbsp; Positive {prob[1]*100:.1f} % / Negative {prob[0]*100:.1f} %</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# XAI section
# ---------------------------------------------------------------------------

def render_xai(user_text: str, model_used: str) -> None:
    st.markdown("#### Explainability")

    tab_lime, tab_shap, tab_compare = st.tabs(["LIME", "SHAP", "Compare models"])

    with tab_lime:
        st.caption(
            "LIME perturbs the input and tracks prediction changes to identify "
            "which words most influence the classification."
        )
        with st.spinner("Computing LIME explanation…"):
            explanation = generate_lime_explanation(user_text, model_used)

            st.markdown("**Feature weights**")
            fig = render_lime_bar(explanation)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("**Highlighted text**")
            html = render_lime_highlighted(explanation, user_text)
            st.markdown(html, unsafe_allow_html=True)

    with tab_shap:
        st.caption(
            "Word-level contributions estimated via coefficient analysis (SVM) "
            "or leave-one-out perturbation (CNN-LSTM)."
        )
        with st.spinner("Computing SHAP explanation…"):
            fig = render_shap_bar(user_text, model_used)
            st.pyplot(fig)
            plt.close(fig)

    with tab_compare:
        st.caption("Run both models on the same input and compare outputs side-by-side.")
        col_cnn, col_svm = st.columns(2)

        with col_cnn:
            st.markdown("**CNN-LSTM**")
            try:
                m, t, p = load_cnn_lstm()
                cnn_pred, cnn_prob, _ = predict_cnn_lstm(user_text, m, t, p)
                label = "Positive" if cnn_pred == 1 else "Negative"
                st.metric("Prediction", label, f"{max(cnn_prob)*100:.1f} %")
                fig = render_shap_bar(user_text, "CNN-LSTM")
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.error(f"CNN-LSTM error: {exc}")

        with col_svm:
            st.markdown("**SVM**")
            try:
                m, tf = load_svm()
                svm_pred, svm_prob, _ = predict_svm(user_text, m, tf)
                label = "Positive" if svm_pred == 1 else "Negative"
                st.metric("Prediction", label, f"{max(svm_prob)*100:.1f} %")
                fig = render_shap_bar(user_text, "SVM")
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.error(f"SVM error: {exc}")


# ---------------------------------------------------------------------------
# Evaluation gallery
# ---------------------------------------------------------------------------

def render_eval_gallery() -> None:
    st.markdown("---")
    st.markdown("#### Evaluation results")

    tab_cnn, tab_svm, tab_compare = st.tabs(["CNN-LSTM", "SVM", "Comparison"])
    panels = [("cnn_lstm", tab_cnn), ("svm", tab_svm)]

    for folder, tab in panels:
        with tab:
            path = os.path.join(RESULTS_DIR, folder)
            if not os.path.exists(path):
                st.info(f"No results yet. Run: `python evaluate.py --model {folder}`")
                continue

            images = sorted(f for f in os.listdir(path) if f.endswith(".png"))
            if not images:
                st.info(f"No plots found. Run: `python evaluate.py --model {folder}`")
                continue

            cols = st.columns(2)
            for i, name in enumerate(images):
                caption = name.replace("_", " ").replace(".png", "").title()
                cols[i % 2].image(os.path.join(path, name), caption=caption)

    with tab_compare:
        path = os.path.join(RESULTS_DIR, "comparison")
        if not os.path.exists(path):
            st.info("No comparison results. Run: `python compare_models.py`")
            return
        images = sorted(f for f in os.listdir(path) if f.endswith(".png"))
        if not images:
            st.info("No comparison plots. Run: `python compare_models.py`")
            return
        cols = st.columns(2)
        for i, name in enumerate(images):
            caption = name.replace("_", " ").replace(".png", "").title()
            cols[i % 2].image(os.path.join(path, name), caption=caption)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    priority = build_sidebar()

    st.title("Sentiment Analysis — Explainable AI")
    st.caption(
        "Dual-pathway architecture (CNN-LSTM · SVM) with LIME and SHAP transparency. "
        "IMDb movie reviews."
    )
    st.divider()

    render_metrics(priority)
    st.markdown("<br>", unsafe_allow_html=True)

    user_text = render_input()

    if st.button("Analyse", type="primary", use_container_width=True):
        if not (user_text and user_text.strip()):
            st.warning("Please enter a review before analysing.")
            return

        with st.spinner("Running inference…"):
            try:
                pred, prob, _cleaned, model_used = route_prediction(user_text, priority)
            except FileNotFoundError as exc:
                st.error(
                    "Model files not found. Train the models first:\n\n"
                    "```\npython train_svm.py\npython train_cnn_lstm.py\n```\n\n"
                    f"Detail: {exc}"
                )
                return
            except Exception as exc:
                st.error(f"Inference error: {exc}")
                return

        render_prediction_result(pred, prob, model_used)
        st.markdown("<br>", unsafe_allow_html=True)
        render_xai(user_text, model_used)

    render_eval_gallery()

    st.divider()
    st.caption("SEM project · CNN-LSTM + SVM · SHAP + LIME · Streamlit")


if __name__ == "__main__":
    main()