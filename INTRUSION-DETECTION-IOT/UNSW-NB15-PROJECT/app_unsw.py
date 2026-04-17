import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='Your TensorFlow version is newer')

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components

# ============================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================
st.set_page_config(page_title="UNSW-NB15 IDS Dashboard", layout="wide", page_icon="🛡️")

CUSTOM_CSS = """
<style>
    /* --- Global --- */
    .main .block-container { padding-top: 1.5rem; }
    
    /* --- Header Banner --- */
    .hero-banner {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-banner h1 { color: #ffffff; margin: 0; font-size: 2rem; }
    .hero-banner p  { color: #c0bfcf; margin: 0.5rem 0 0 0; font-size: 1rem; }
    
    /* --- Metric Cards --- */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #8888aa;
        margin-top: 0.3rem;
    }
    
    /* --- Detection Result Cards --- */
    .result-safe {
        background: linear-gradient(135deg, #0d2818, #1a4d2e);
        border: 1px solid #2ecc71;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-safe h2 { color: #2ecc71; margin: 0; }
    
    .result-danger {
        background: linear-gradient(135deg, #2d0a0a, #4d1a1a);
        border: 1px solid #e74c3c;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-danger h2 { color: #e74c3c; margin: 0; }
    
    /* --- Section Headers --- */
    .section-header {
        background: linear-gradient(90deg, rgba(102,126,234,0.15), transparent);
        padding: 0.6rem 1rem;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_resources():
    """Load model, preprocessors, and initialize explainers (cached)."""
    model = tf.keras.models.load_model('dnn_model_unsw.keras')
    
    with open('preprocessors_unsw.pkl', 'rb') as f:
        data = pickle.load(f)
    
    scaler = data['scaler']
    feature_names = data['feature_names']
    background_data = data['background_data']
    encoders = data['encoders']
    
    # LIME explainer
    lime_explainer = LimeTabularExplainer(
        background_data,
        feature_names=feature_names,
        class_names=list(encoders['target'].classes_),
        mode='classification',
        discretize_continuous=True
    )
    
    # SHAP explainer
    try:
        shap_explainer = shap.GradientExplainer(model, background_data[:100])
    except Exception as e:
        st.warning(f"SHAP GradientExplainer init issue: {e}")
        shap_explainer = None
    
    return model, scaler, feature_names, background_data, encoders, lime_explainer, shap_explainer


def predict(model, input_data, encoders):
    """Run model prediction and return structured results."""
    probs = model.predict(input_data, verbose=0)
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label_name = encoders['target'].inverse_transform([predicted_class])[0]
    return {
        'probs': probs,
        'class_idx': predicted_class,
        'confidence': confidence,
        'label': label_name,
        'is_normal': label_name.lower() == 'normal'
    }


def generate_lime_html(lime_explainer, input_sample, model, num_features=10):
    """Generate LIME explanation HTML with custom styling for readability."""
    exp = lime_explainer.explain_instance(
        input_sample,
        lambda x: model.predict(x, verbose=0),
        num_features=num_features,
        top_labels=1
    )
    
    html = exp.as_html()
    inject_css = """
    <style>
        body, body * { background-color: #ffffff !important; color: #000000 !important; }
        table.table-condensed { background-color: #ffffff !important; border-color: #ddd !important; }
        table.table-condensed td { color: #000 !important; background-color: #fff !important; }
        table.table-condensed th { color: #000 !important; background-color: #e8e8e8 !important; }
        text { fill: #000000 !important; }
    </style>
    """
    return html.replace('<html>', f'<html>{inject_css}')


def generate_shap_figure(shap_explainer, input_data, feature_names, predicted_class, label_name):
    """Generate SHAP bar chart as a matplotlib figure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = shap_explainer.shap_values(input_data)
    
    if isinstance(shap_values, list):
        sv = shap_values[predicted_class]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, predicted_class]
    else:
        sv = shap_values

    importance = sv[0]
    top_n = min(10, len(feature_names))
    indices = np.argsort(np.abs(importance))[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in importance[indices]]
    bars = ax.barh(range(len(indices)), importance[indices], color=colors, edgecolor='none', height=0.7)
    
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10, color='#cccccc')
    ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=11, color='#cccccc')
    ax.set_title(f'Top {top_n} Feature Contributions → {label_name.upper()}',
                 fontsize=13, fontweight='bold', color='#ffffff', pad=12)
    ax.axvline(x=0, color='#555555', linestyle='-', linewidth=1)
    
    for bar, val in zip(bars, importance[indices]):
        w = bar.get_width()
        ax.text(w + (0.005 if w > 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f'{w:.4f}', va='center',
                ha='left' if w > 0 else 'right',
                fontsize=9, color='#bbbbbb')
    
    plt.tight_layout()
    return fig


# ============================================================
# LOAD RESOURCES
# ============================================================
try:
    model, scaler, feature_names, background_data, encoders, lime_explainer, shap_explainer = load_resources()
    num_classes = len(encoders['target'].classes_)
    num_features = len(feature_names)
except Exception as e:
    st.error(f"**Failed to load resources:** {e}")
    st.info("Make sure to run `python train_unsw.py` first to train the UNSW model and generate preprocessors.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🛡️ System Info")
    st.caption(f"**Model:** DNN (3×64 + Dropout)")
    st.caption(f"**Backend:** TensorFlow {tf.__version__}")
    st.caption(f"**Dataset:** UNSW-NB15")
    st.caption(f"**Features:** {num_features}")
    st.caption(f"**Classes:** {num_classes}")
    
    st.divider()
    
    st.markdown("### 📋 Target Classes")
    for i, cls in enumerate(encoders['target'].classes_):
        icon = "🟢" if cls.lower() == "normal" else "🔴"
        st.caption(f"{icon} {cls}")
    
    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "This dashboard visualizes network traffic classification using a Deep Neural Network trained on the "
        "UNSW-NB15 dataset (5-class variant. Explainability is provided via SHAP and LIME."
    )


# ============================================================
# HERO BANNER
# ============================================================
st.markdown("""
<div class="hero-banner">
    <h1>🛡️ UNSW-NB15 Intrusion Detection System</h1>
    <p>Transparent IoT security monitoring powered by Explainable AI</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# METRICS ROW
# ============================================================
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{num_features}</div>
        <div class="metric-label">Input Features</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{num_classes}</div>
        <div class="metric-label">Target Classes</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(background_data)}</div>
        <div class="metric-label">Background Samples</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">DNN</div>
        <div class="metric-label">Active Model</div>
    </div>""", unsafe_allow_html=True)


# ============================================================
# 1. INPUT SECTION
# ============================================================
st.markdown('<div class="section-header">📡 Network Traffic Input</div>', unsafe_allow_html=True)

if 'current_sample' not in st.session_state:
    st.session_state.current_sample = background_data[0].copy()

col_input, col_preview = st.columns([1, 2])

with col_input:
    st.info("Select a random known sample from the dataset or adjust feature values manually.")
    
    if st.button("🔀 Generate Random Sample", use_container_width=True):
        idx = np.random.randint(0, len(background_data))
        new_sample = background_data[idx].copy()
        st.session_state.current_sample = new_sample
        
        display_count = min(8, num_features)
        for i in range(display_count):
            st.session_state[f"val_{i}"] = float(new_sample[i])
    
    input_mode = st.radio("Input Mode:", ["Use Selected Sample", "Manual Adjustment"], horizontal=True)

with col_preview:
    current_vals = st.session_state.current_sample
    updated_vals = list(current_vals)
    
    # Show adjustable sliders for top features (limit to 8 for UI cleanlyness)
    display_count = min(8, num_features)
    slider_cols = st.columns(2)
    
    for i in range(display_count):
        feat = feature_names[i]
        with slider_cols[i % 2]:
            if input_mode == "Manual Adjustment":
                updated_vals[i] = st.slider(feat, 0.0, 1.0, float(current_vals[i]), key=f"val_{i}")
            else:
                st.slider(feat, 0.0, 1.0, float(current_vals[i]), disabled=True, key=f"val_{i}")
    
    final_input = np.array(updated_vals).reshape(1, -1)

# Expandable table with all feature values
with st.expander("📊 View All 38 Feature Values"):
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': [f"{v:.4f}" for v in updated_vals]
    })
    chunk = len(feature_names) // 3 + 1
    fc1, fc2, fc3 = st.columns(3)
    with fc1: st.dataframe(feat_df.iloc[:chunk], use_container_width=True, hide_index=True)
    with fc2: st.dataframe(feat_df.iloc[chunk:2*chunk], use_container_width=True, hide_index=True)
    with fc3: st.dataframe(feat_df.iloc[2*chunk:], use_container_width=True, hide_index=True)


# ============================================================
# 2. DETECTION & XAI
# ============================================================
st.markdown('<div class="section-header">🔍 Deep Inspection</div>', unsafe_allow_html=True)

if st.button("⚡ Analyze Traffic Pattern", type="primary", use_container_width=True):
    
    # --- Prediction ---
    result = predict(model, final_input, encoders)
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if result['is_normal']:
            st.markdown(f"""
            <div class="result-safe">
                <h2>✅ NORMAL</h2>
                <p style="color:#8fdfb0; margin-top:0.5rem;">
                    Traffic pattern appears legitimate. No attack signatures found.<br>
                    <strong>Confidence: {result['confidence']*100:.1f}%</strong>
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-danger">
                <h2>🚨 {result['label'].upper()} ATTACK DETECTED</h2>
                <p style="color:#f5a5a5; margin-top:0.5rem;">
                    Malicious activity patterns match known signatures!<br>
                    <strong>Confidence: {result['confidence']*100:.1f}%</strong>
                </p>
            </div>""", unsafe_allow_html=True)
    
    with res_col2:
        st.markdown("**Probability Matrix**")
        prob_df = pd.DataFrame({
            'Class': encoders['target'].classes_,
            'Probability': result['probs'][0]
        }).sort_values('Probability', ascending=False)
        st.bar_chart(prob_df.set_index('Class'), height=250)
    
    # --- Explanations ---
    st.markdown('<div class="section-header">🧠 Explainable AI Insights</div>', unsafe_allow_html=True)
    tab_lime, tab_shap = st.tabs(["🔍 LIME — Local Feature Impact", "🧬 SHAP — Feature Contributions"])
    
    with tab_lime:
        st.markdown("LIME creates a **local surrogate model** to identify which features had the most impact on **this specific** prediction.")
        with st.spinner("Generating LIME explanation..."):
            try:
                lime_html = generate_lime_html(lime_explainer, final_input[0], model)
                components.html(lime_html, height=500, scrolling=True)
            except Exception as e:
                st.error(f"LIME error: {e}")
    
    with tab_shap:
        st.markdown("SHAP values quantify how much each feature **shifted** the prediction away from the average baseline.")
        with st.spinner("Calculating SHAP values..."):
            try:
                if shap_explainer is not None:
                    fig = generate_shap_figure(shap_explainer, final_input, feature_names, result['class_idx'], result['label'])
                    st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.error(f"SHAP error: {e}")

else:
    st.info("👆 Click **Analyze Traffic Pattern** to run the Deep Neural Network Model and XAI explainers.")
