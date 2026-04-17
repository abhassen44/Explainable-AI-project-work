import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="CHEFS Smart Energy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #5865F2;
        padding: 1rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 15px rgba(88, 101, 242, 0.3);
    }
    
    div[data-testid="stMetric"] label {
        color: rgba(255,255,255,0.8) !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.8rem !important;
    }
    
    /* Success metric (green) */
    .success-metric div[data-testid="stMetric"] {
        background: #2ecc71;
    }
    
    /* Warning metric (orange) */
    .warning-metric div[data-testid="stMetric"] {
        background: #e74c3c;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background: #f5f7fa;
        border-left: 4px solid #5865F2;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Header styling */
    .main-header {
        color: #5865F2;
        font-size: 2.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD ARTIFACTS (The "Brain")
# ==========================================
@st.cache_resource
def load_system():
    data = pd.read_csv("simulation_data.csv")
    feature_names = data.drop("Actual_kW", axis=1).columns.tolist()
    
    x_scaler = joblib.load('x_scaler.pkl')
    y_scaler = joblib.load('y_scaler.pkl')
    
    input_dim = len(feature_names)
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    model.load_state_dict(torch.load("chefs_model.pth", map_location=torch.device('cpu')))
    model.eval()
    
    return model, data, feature_names, x_scaler, y_scaler

model, df, feature_names, x_scaler, y_scaler = load_system()

# Feature name mapping for better readability
FEATURE_LABELS = {
    'Global_active_power': '🔌 Active Power',
    'Global_reactive_power': '⚡ Reactive Power',
    'Voltage': '🔋 Voltage',
    'Global_intensity': '💡 Intensity',
    'Sub_metering_1': '🏠 Kitchen',
    'Sub_metering_2': '🧺 Laundry',
    'Sub_metering_3': '❄️ Climate Control',
    'hour': '🕐 Hour',
    'weekday': '📅 Day of Week',
    'month': '📆 Month'
}

def get_friendly_name(feature):
    return FEATURE_LABELS.get(feature, feature)

# ==========================================
# 3. HEADER SECTION
# ==========================================
col_header1, col_header2 = st.columns([3, 1])

with col_header1:
    st.markdown('<p class="main-header">⚡ CHEFS: Smart Energy Forecasting</p>', unsafe_allow_html=True)
    st.caption("Intelligent Energy Prediction with Explainable AI")

with col_header2:
    st.markdown("""
    <div style="text-align: right; padding: 10px;">
        <span style="background: #2ecc71; padding: 5px 15px; border-radius: 20px; color: white; font-weight: 600;">
            🟢 ONLINE
        </span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==========================================
# 4. SIDEBAR: ENHANCED CONTROL PANEL
# ==========================================
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown("---")
    
    # Time navigation with better UX
    st.markdown("### ⏱️ Time Navigation")
    
    # Quick navigation buttons
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    
    if 'hour_idx' not in st.session_state:
        st.session_state.hour_idx = 0
    
    with col_nav1:
        if st.button("⏮️ Start", use_container_width=True):
            st.session_state.hour_idx = 0
    with col_nav2:
        if st.button("⏭️ End", use_container_width=True):
            st.session_state.hour_idx = len(df) - 1
    with col_nav3:
        if st.button("🎲 Random", use_container_width=True):
            st.session_state.hour_idx = np.random.randint(0, len(df))
    
    hour_idx = st.slider(
        "Select Time Point",
        min_value=0,
        max_value=len(df)-1,
        value=st.session_state.hour_idx,
        key="time_slider",
        help="Slide to navigate through the last 200 hours of data"
    )
    st.session_state.hour_idx = hour_idx
    
    # Show time context
    st.info(f"📍 Viewing hour **{hour_idx + 1}** of **{len(df)}**")
    
    st.markdown("---")
    
    # Display settings
    st.markdown("### ⚙️ Display Settings")
    
    show_trend = st.checkbox("📈 Show Trend Chart", value=True, help="Display prediction vs actual trend")
    show_features = st.checkbox("📊 Show Feature Values", value=True, help="Display current sensor readings")
    auto_explain = st.checkbox("🤖 Auto-Generate Explanations", value=False, help="Automatically run XAI on time change")
    
    st.markdown("---")
    
    # Model info
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        **Architecture:** Neural Network
        - Input Layer: 10 features
        - Hidden Layer 1: 128 neurons (ReLU)
        - Hidden Layer 2: 64 neurons (ReLU)
        - Output: 1 (Power in kW)
        
        **Training Data:** Household Power Consumption
        **Update Frequency:** Hourly predictions
        """)

# Extract the specific row of data
row = df.iloc[hour_idx]
actual_kw = row['Actual_kW']
features_raw = row.drop("Actual_kW").values.reshape(1, -1)
features_tensor = torch.tensor(features_raw, dtype=torch.float32)

# ==========================================
# 5. PREDICTION SECTION
# ==========================================
st.markdown("### 📊 Real-Time Predictions")

# Make Prediction
with torch.no_grad():
    pred_scaled = model(features_tensor).item()
    data_min = y_scaler.data_min_[0]
    data_max = y_scaler.data_max_[0]
    pred_kw = pred_scaled * (data_max - data_min) + data_min

error = pred_kw - actual_kw
error_pct = abs(error / actual_kw) * 100 if actual_kw != 0 else 0

# Determine status based on error
if abs(error) < 0.1:
    status_emoji = "✅"
    status_text = "Excellent"
    status_color = "success"
elif abs(error) < 0.3:
    status_emoji = "⚠️"
    status_text = "Good"
    status_color = "warning"
else:
    status_emoji = "❌"
    status_text = "Needs Review"
    status_color = "error"

# Metric cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Predicted Power",
        value=f"{pred_kw:.3f} kW",
        help="AI model's prediction for power consumption"
    )

with col2:
    st.metric(
        label="📏 Actual Power",
        value=f"{actual_kw:.3f} kW",
        help="Ground truth measurement from sensors"
    )

with col3:
    delta_color = "inverse" if error > 0 else "normal"
    st.metric(
        label="📉 Prediction Error",
        value=f"{error:.3f} kW",
        delta=f"{error_pct:.1f}%",
        delta_color=delta_color,
        help="Difference between predicted and actual values"
    )

with col4:
    st.metric(
        label=f"{status_emoji} Status",
        value=status_text,
        help="Overall prediction quality assessment"
    )

# ==========================================
# 6. TREND VISUALIZATION
# ==========================================
if show_trend:
    st.markdown("### 📈 Prediction Trend Analysis")
    
    # Calculate all predictions for the dataset
    @st.cache_data
    def get_all_predictions():
        all_features = torch.tensor(df.drop("Actual_kW", axis=1).values, dtype=torch.float32)
        with torch.no_grad():
            all_preds_scaled = model(all_features).numpy()
        all_preds = all_preds_scaled * (data_max - data_min) + data_min
        return all_preds.flatten()
    
    all_preds = get_all_predictions()
    all_actual = df['Actual_kW'].values
    
    # Create interactive Plotly chart
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Power Consumption: Predicted vs Actual", "Prediction Error")
    )
    
    # Main trend lines
    fig.add_trace(
        go.Scatter(x=list(range(len(all_actual))), y=all_actual, 
                   name="Actual", line=dict(color="#3498db", width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(all_preds))), y=all_preds, 
                   name="Predicted", line=dict(color="#e74c3c", width=2, dash="dot")),
        row=1, col=1
    )
    
    # Current position marker
    fig.add_trace(
        go.Scatter(x=[hour_idx], y=[actual_kw], 
                   name="Current Point", mode="markers",
                   marker=dict(size=15, color="#2ecc71", symbol="star")),
        row=1, col=1
    )
    
    # Error chart
    errors = all_preds - all_actual
    colors = ['#2ecc71' if e >= 0 else '#e74c3c' for e in errors]
    fig.add_trace(
        go.Bar(x=list(range(len(errors))), y=errors, 
               name="Error", marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    fig.update_layout(
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(title_text="Time Index (Hours)", row=2, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Error (kW)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 7. CURRENT FEATURE VALUES
# ==========================================
if show_features:
    st.markdown("### 🔍 Current Sensor Readings")
    
    feature_cols = st.columns(5)
    feature_values = row.drop("Actual_kW")
    
    for i, (fname, fval) in enumerate(feature_values.items()):
        col_idx = i % 5
        with feature_cols[col_idx]:
            friendly_name = get_friendly_name(fname)
            st.markdown(f"""
            <div style="background: #f5f7fa; 
                        padding: 15px; border-radius: 10px; text-align: center; margin: 5px 0;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="font-size: 0.8rem; color: #666; margin-bottom: 5px;">{friendly_name}</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #2c3e50;">{fval:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ==========================================
# 8. XAI LAYER (Enhanced)
# ==========================================
st.markdown("### 🧠 Explainable AI Analysis")
st.markdown("Understand *why* the AI made this prediction using two complementary explanation methods.")

xai_tab1, xai_tab2, xai_tab3 = st.tabs(["🔷 SHAP Analysis", "🟢 LIME Analysis", "📚 Compare Methods"])

# --- SHAP TAB ---
with xai_tab1:
    col_shap1, col_shap2 = st.columns([2, 1])
    
    with col_shap1:
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** distributes the prediction fairly among all features 
        using concepts from game theory. Each feature gets credit proportional to its contribution.
        """)
    
    with col_shap2:
        run_shap = st.button("🔷 Generate SHAP Explanation", key="shap_btn", use_container_width=True)
    
    if run_shap or (auto_explain and 'last_shap_idx' in st.session_state and st.session_state.last_shap_idx != hour_idx):
        st.session_state.last_shap_idx = hour_idx
        
        with st.spinner("🔄 Running SHAP analysis..."):
            background = torch.tensor(df.drop("Actual_kW", axis=1).values[:50], dtype=torch.float32)
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(features_tensor)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            sv = shap_values.flatten()
            
            # Create interactive Plotly bar chart
            fig = go.Figure()
            
            colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in sv]
            friendly_names = [get_friendly_name(f) for f in feature_names]
            
            # Sort by absolute value
            sorted_indices = np.argsort(np.abs(sv))[::-1]
            sorted_names = [friendly_names[i] for i in sorted_indices]
            sorted_values = [sv[i] for i in sorted_indices]
            sorted_colors = [colors[i] for i in sorted_indices]
            
            fig.add_trace(go.Bar(
                y=sorted_names,
                x=sorted_values,
                orientation='h',
                marker_color=sorted_colors,
                text=[f"{v:.4f}" for v in sorted_values],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="SHAP Feature Contributions",
                xaxis_title="Impact on Prediction (kW)",
                yaxis_title="",
                height=400,
                showlegend=False,
                margin=dict(l=150, r=50, t=50, b=50)
            )
            
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            max_idx = np.argmax(np.abs(sv))
            max_feature = get_friendly_name(feature_names[max_idx])
            direction = "⬆️ INCREASE" if sv[max_idx] > 0 else "⬇️ DECREASE"
            
            col_ins1, col_ins2 = st.columns(2)
            with col_ins1:
                st.success(f"**Top Driver:** {max_feature} → {direction}")
            with col_ins2:
                st.info(f"**Total Positive Effect:** {sum(v for v in sv if v > 0):.4f} kW")

# --- LIME TAB ---
with xai_tab2:
    col_lime1, col_lime2 = st.columns([2, 1])
    
    with col_lime1:
        st.markdown("""
        **LIME (Local Interpretable Model-agnostic Explanations)** creates a simple local approximation 
        of the model around the specific prediction point, showing how small changes affect the output.
        """)
    
    with col_lime2:
        run_lime = st.button("🟢 Generate LIME Explanation", key="lime_btn", use_container_width=True)
    
    if run_lime or (auto_explain and 'last_lime_idx' in st.session_state and st.session_state.last_lime_idx != hour_idx):
        st.session_state.last_lime_idx = hour_idx
        
        with st.spinner("🔄 Running LIME analysis..."):
            training_data = df.drop("Actual_kW", axis=1).values
            
            def predict_fn(X):
                X_tensor = torch.tensor(X, dtype=torch.float32)
                with torch.no_grad():
                    preds = model(X_tensor).numpy()
                return preds.flatten()
            
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                mode='regression',
                verbose=False
            )
            
            lime_exp = lime_explainer.explain_instance(
                features_raw.flatten(),
                predict_fn,
                num_features=len(feature_names)
            )
            
            lime_weights = dict(lime_exp.as_list())
            
            # Create Plotly chart
            fig = go.Figure()
            
            sorted_features = sorted(lime_weights.items(), key=lambda x: abs(x[1]), reverse=True)
            labels = [f[0] for f in sorted_features]
            values = [f[1] for f in sorted_features]
            colors = ['#27ae60' if v > 0 else '#c0392b' for v in values]
            
            fig.add_trace(go.Bar(
                y=labels,
                x=values,
                orientation='h',
                marker_color=colors,
                text=[f"{v:.4f}" for v in values],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="LIME Feature Weights (Local Model)",
                xaxis_title="Feature Weight",
                yaxis_title="",
                height=400,
                showlegend=False,
                margin=dict(l=200, r=50, t=50, b=50)
            )
            
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Local Prediction", f"{lime_exp.local_pred[0]:.4f}")
            with col_m2:
                st.metric("Model R² Score", f"{lime_exp.score:.4f}")
            with col_m3:
                if sorted_features:
                    top_feature = sorted_features[0][0]
                    st.metric("Top Feature", top_feature[:20] + "..." if len(top_feature) > 20 else top_feature)

# --- COMPARISON TAB ---
with xai_tab3:
    st.markdown("""
    ### 🔬 SHAP vs LIME: When to Use Which?
    
    | Aspect | SHAP 🔷 | LIME 🟢 |
    |--------|---------|---------|
    | **Approach** | Game theory (Shapley values) | Local surrogate model |
    | **Consistency** | Always consistent | May vary with perturbations |
    | **Global View** | Yes, aggregatable | Primarily local |
    | **Speed** | Slower for complex models | Generally faster |
    | **Best For** | Understanding overall model | Explaining specific predictions |
    
    ---
    
    **💡 Recommendation:** Use **SHAP** when you need consistent, theoretically grounded explanations. 
    Use **LIME** for quick, intuitive local insights about specific predictions.
    """)

# ==========================================
# 9. FOOTER: RAW DATA & EXPORT
# ==========================================
st.divider()

col_foot1, col_foot2 = st.columns([2, 1])

with col_foot1:
    with st.expander("📋 View Raw Sensor Data"):
        display_df = pd.DataFrame([row.drop("Actual_kW")], columns=feature_names)
        display_df.columns = [get_friendly_name(c) for c in feature_names]
        st.dataframe(display_df, use_container_width=True)

with col_foot2:
    with st.expander("📥 Export Options"):
        st.download_button(
            label="📄 Download Current Data (CSV)",
            data=pd.DataFrame([row]).to_csv(index=False),
            file_name=f"energy_data_hour_{hour_idx}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        st.caption("CHEFS Framework v2.0 | Powered by PyTorch & Streamlit")