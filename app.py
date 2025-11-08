# ==========================================
# Predictive Maintenance Dashboard
# Developed by: The Vanguards
# Members: Pravin | Aiteswar | Ashwien
# Fully Automatic Version: Correct SHAP for Multi-Class with Progress Bar
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import shap

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔧 Predictive Maintenance Dashboard for Industrial Equipment")
st.markdown("### A data-driven approach to anticipate failures and optimize maintenance")

# ------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------
menu = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard Overview",
        "📊 Sensor Data Visualization",
        "🧠 Failure Prediction",
        "📈 Feature Insights",
        "🛠️ Maintenance Log",
        "📉 Model Performance"
    ]
)

# ------------------------------------------
# LOAD / UPLOAD DATA SECTION
# ------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

st.sidebar.header("📂 Upload Sensor Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.sidebar.success("✅ Data uploaded successfully!")

    # Define sensor columns
    sensor_cols = [c for c in data.columns if c not in ['timestamp', 'status', 'predicted_status']]

    # ------------------------------------------
    # AUTOMATIC MODEL TRAINING AND PREDICTION
    # ------------------------------------------
    if 'status' in data.columns and sensor_cols:
        X = data[sensor_cols]
        y = data['status']

        # Encode status labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        st.session_state.label_encoder = le

        # Train RandomForest
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y_encoded)
        st.session_state.model = model

        # Predict
        predictions_encoded = model.predict(X)
        data['predicted_status'] = le.inverse_transform(predictions_encoded)

else:
    st.sidebar.warning("Please upload a dataset to continue.")
    st.stop()

# ------------------------------------------
# 1. DASHBOARD OVERVIEW
# ------------------------------------------
if menu == "🏠 Dashboard Overview":
    st.subheader("System Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Number of Sensors", len(sensor_cols))
    with col3:
        st.metric("Time Range", f"{data['timestamp'].min().date()} → {data['timestamp'].max().date()}")

    st.markdown("### Equipment Health Summary")
    if 'status' in data.columns:
        fig = px.pie(data, names='status', title="Current Equipment Status Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'status' column found — please include a status label (e.g., Normal/Warning/Critical).")

# ------------------------------------------
# 2. SENSOR DATA VISUALIZATION
# ------------------------------------------
elif menu == "📊 Sensor Data Visualization":
    st.subheader("Sensor Trend Analysis")
    if sensor_cols:
        selected_sensor = st.selectbox("Select Sensor", sensor_cols)
        fig = px.line(data, x='timestamp', y=selected_sensor, title=f"{selected_sensor} Over Time")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Correlation Heatmap")
        st.write(px.imshow(data[sensor_cols].corr(), color_continuous_scale='Blues'))
    else:
        st.warning("⚠️ No numeric sensor columns detected.")

# ------------------------------------------
# 3. FAILURE PREDICTION (AUTOMATIC)
# ------------------------------------------
elif menu == "🧠 Failure Prediction":
    st.subheader("Predicted Equipment Health Status")
    st.success("✅ Predictions generated automatically upon CSV upload!")
    fig = px.histogram(data, x='predicted_status', title="Predicted Equipment Status")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "Download Prediction Results",
        data.to_csv(index=False).encode('utf-8'),
        "predictions.csv",
        "text/csv"
    )

# ------------------------------------------
# 4. FEATURE INSIGHTS (SHAP VALUES WITH PROGRESS BAR)
# ------------------------------------------
elif menu == "📈 Feature Insights":
    st.subheader("Model Explainability - SHAP Insights")
    st.info("This section shows which features (sensors) most influence failure predictions.")

    model = st.session_state.model
    sample_X = data[sensor_cols].sample(min(200, len(data)))

    with st.spinner("Calculating SHAP values..."):
        explainer = shap.TreeExplainer(model)
        
        # Initialize progress bar
        progress_bar = st.progress(0)
        total_rows = len(sample_X)
        shap_values_list = []

        # Compute SHAP values row by row to update progress
        for i, (_, row) in enumerate(sample_X.iterrows()):
            shap_val = explainer.shap_values(row)
            shap_values_list.append(shap_val)
            progress_bar.progress((i + 1) / total_rows)

        progress_bar.empty()  # remove progress bar when done

        # Handle multi-class SHAP values
        if isinstance(shap_values_list[0], list):  # multi-class
            # Stack per class
            class_arrays = []
            for class_idx in range(len(shap_values_list[0])):
                class_arrays.append(np.array([shap_val[class_idx] for shap_val in shap_values_list]))
            # Compute mean absolute across classes
            mean_abs_shap = np.mean([np.abs(ca).mean(axis=0) for ca in class_arrays], axis=0)
        else:  # single-class
            mean_abs_shap = np.abs(np.vstack(shap_values_list)).mean(axis=0)

        shap_summary = pd.DataFrame({
            'Feature': sample_X.columns,
            'Mean |SHAP value|': mean_abs_shap
        }).sort_values(by='Mean |SHAP value|', ascending=False)

        st.markdown("### Feature Importance Based on SHAP Values")
        st.dataframe(shap_summary)

        # Plotly bar chart
        fig = px.bar(shap_summary, x='Feature', y='Mean |SHAP value|', title="SHAP Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# 5. MAINTENANCE LOG
# ------------------------------------------
elif menu == "🛠️ Maintenance Log":
    st.subheader("Maintenance Alerts and Actions")
    st.markdown("""
    | Date | Machine | Alert Level | Recommended Action |
    |-------|----------|-------------|--------------------|
    | 2025-11-01 | Motor #2 | ⚠️ Warning | Schedule inspection for vibration anomaly |
    | 2025-11-03 | Pump #1 | 🔴 Critical | Immediate shutdown recommended |
    | 2025-11-05 | Bearing #4 | 🟢 Normal | No action required |
    """)
    st.download_button("Download Maintenance Log", "Sample maintenance log.", "maintenance_log.txt")

# ------------------------------------------
# 6. MODEL PERFORMANCE
# ------------------------------------------
elif menu == "📉 Model Performance":
    st.subheader("Model Evaluation Metrics")
    y_true_encoded = st.session_state.label_encoder.transform(data['status'])
    y_pred_encoded = st.session_state.label_encoder.transform(data['predicted_status'])

    st.text("Classification Report:")
    st.text(classification_report(y_true_encoded, y_pred_encoded, target_names=st.session_state.label_encoder.classes_))

    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual"),
                    x=st.session_state.label_encoder.classes_,
                    y=st.session_state.label_encoder.classes_)
    st.plotly_chart(fig, use_container_width=True)

    if len(st.session_state.label_encoder.classes_) == 2:
        fpr, tpr, _ = roc_curve(y_true_encoded, y_pred_encoded)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(dash='dash')))
        fig_roc.update_layout(title=f"ROC Curve (AUC = {roc_auc:.2f})",
                              xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig_roc, use_container_width=True)

# ------------------------------------------
# END
# ------------------------------------------
st.sidebar.markdown("---")
st.sidebar.info("Developed by **The Vanguards** | Streamlit Predictive Maintenance Project")
