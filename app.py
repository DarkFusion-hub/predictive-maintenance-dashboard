# ==========================================
# Predictive Maintenance Dashboard - Advanced
# Developed by: The Vanguards
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

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔧 Predictive Maintenance Dashboard")
st.markdown("*Data-driven insights to prevent failures, optimize maintenance, and improve operational efficiency*")

# ------------------------------
# UPLOAD / LOAD DATA
# ------------------------------
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Handle timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'DateTime' in df.columns:
        df.rename(columns={'DateTime': 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'time' in df.columns:
        df.rename(columns={'time': 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        st.warning("⚠ No timestamp column found. Using default sequence index as timestamp.")
        df['timestamp'] = pd.date_range(start='2025-01-01', periods=len(df), freq='H')
    return df

st.sidebar.header("📂 Upload Sensor Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file:
    data = load_data(uploaded_file)
    st.sidebar.success("✅ Data uploaded successfully!")
    sensor_cols = [c for c in data.columns if c not in ['timestamp', 'status', 'predicted_status']]

    # ------------------------------
    # TRAIN MODEL
    # ------------------------------
    if 'status' in data.columns and sensor_cols:
        X = data[sensor_cols].copy()
        y = data['status']

        # Encode categorical sensor columns automatically
        for col in X.select_dtypes(include='object').columns:
            le_col = LabelEncoder()
            X[col] = le_col.fit_transform(X[col])

        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        st.session_state.label_encoder = le

        # Train RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y_encoded)
        st.session_state.model = model

        pred_encoded = model.predict(X)
        data['predicted_status'] = le.inverse_transform(pred_encoded)

        # Define risk levels
        data['risk_level'] = np.where(data['predicted_status'] == 'Critical', 'High',
                                      np.where(data['predicted_status'] == 'Warning', 'Medium', 'Low'))
else:
    st.sidebar.warning("Please upload a dataset to continue.")
    st.stop()

# ------------------------------
# TABS NAVIGATION
# ------------------------------
tabs = st.tabs([
    "🏠 Overview",
    "📊 Sensor Trends",
    "🧠 Failure Prediction",
    "📈 SHAP Insights",
    "⚡ What-If Scenarios",
    "💡 Recommendations",
    "📜 Outcome & Conclusion",
    "🛠 Maintenance Log",
    "📉 Model Performance"
])

# ------------------------------
# 1. DASHBOARD OVERVIEW
# ------------------------------
with tabs[0]:
    st.subheader("System Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(data))
    col2.metric("Number of Sensors", len(sensor_cols))
    col3.metric("Time Range", f"{data['timestamp'].min().date()} → {data['timestamp'].max().date()}")

    st.markdown("### Equipment Health Distribution")
    if 'status' in data.columns:
        fig = px.pie(data, names='status', title="Current Equipment Status", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# 2. SENSOR TRENDS
# ------------------------------
with tabs[1]:
    st.subheader("Interactive Sensor Trends")
    sensors_selected = st.multiselect("Select Sensors", sensor_cols, default=sensor_cols[:2])
    if sensors_selected:
        fig = px.line(data, x='timestamp', y=sensors_selected, title="Sensor Readings Over Time")
        fig.update_layout(xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        corr = data[sensors_selected].corr()
        st.markdown("#### Sensor Correlation Heatmap")
        st.write(px.imshow(corr, color_continuous_scale='Blues'))

# ------------------------------
# 3. FAILURE PREDICTION
# ------------------------------
with tabs[2]:
    st.subheader("Failure Prediction & Risk Alerts")
    if 'predicted_status' in data.columns:
        fig = px.histogram(data, x='predicted_status', color='risk_level',
                           title="Predicted Equipment Health",
                           barmode='group',
                           color_discrete_map={'High':'red','Medium':'orange','Low':'green'})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### High-Risk Equipment")
        high_risk = data[data['risk_level'] == 'High']
        if not high_risk.empty:
            st.dataframe(high_risk[['timestamp','predicted_status','risk_level']])
        else:
            st.info("No high-risk equipment detected currently.")

        st.download_button("Download Predictions", data.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")

# ------------------------------
# 4. FEATURE INSIGHTS (SHAP)
# ------------------------------
with tabs[3]:
    st.subheader("Feature Importance - SHAP Values")
    if 'model' in st.session_state and sensor_cols:
        model = st.session_state.model
        sample_X = data[sensor_cols].sample(min(200,len(data)))
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(sample_X)
            shap_vals = shap_values.values
            if shap_vals.ndim == 2:
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            elif shap_vals.ndim == 3:
                mean_abs_shap = np.abs(shap_vals).mean(axis=(0,2))
            shap_summary = pd.DataFrame({'Feature': sample_X.columns, 'Mean |SHAP|': mean_abs_shap}).sort_values(by='Mean |SHAP|', ascending=False)
            st.dataframe(shap_summary)
            fig = px.bar(shap_summary, x='Feature', y='Mean |SHAP|', title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# 5. WHAT-IF SCENARIOS
# ------------------------------
with tabs[4]:
    st.subheader("What-If Scenarios")
    st.info("Simulate changes in sensor readings to see potential impact on predicted status.")
    scenario_sensor = st.selectbox("Select Sensor to Modify", sensor_cols)
    scenario_value = st.slider(f"Adjust {scenario_sensor} Value", float(data[scenario_sensor].min()), float(data[scenario_sensor].max()), float(data[scenario_sensor].mean()))
    
    scenario_X = X.copy()
    scenario_X[scenario_sensor] = scenario_value
    scenario_pred = st.session_state.model.predict(scenario_X)
    scenario_status = st.session_state.label_encoder.inverse_transform(scenario_pred)
    
    st.markdown(f"*Predicted Status after adjusting {scenario_sensor} = {scenario_value}:*")
    st.write(pd.Series(scenario_status).value_counts())

# ------------------------------
# 6. RECOMMENDATIONS
# ------------------------------
with tabs[5]:
    st.subheader("Solutions / Recommendations")
    st.markdown("""
    - *High-risk equipment:* Schedule immediate inspection or preventive maintenance.
    - *Medium-risk equipment:* Monitor sensor trends closely and prepare contingency.
    - *Low-risk equipment:* Maintain normal operation and routine checks.
    - *Feature insights:* Focus maintenance on components with highest SHAP values.
    - *Trend analysis:* Adjust operational parameters that contribute to high vibration, temperature, or power usage.
    """)

# ------------------------------
# 7. OUTCOME & CONCLUSION
# ------------------------------
with tabs[6]:
    st.subheader("Objective & Outcome")
    st.markdown("""
    *Objective:* Develop a predictive maintenance system to reduce failures and optimize operations.
    
    *Key Achievements:*
    - Built interactive dashboards for real-time/historical sensor analysis.
    - Developed ML-based predictions for equipment health.
    - Identified high-risk components and actionable insights.
    - SHAP analysis explained key contributing factors to failures.
    - What-if scenarios allow proactive maintenance planning.
    
    *Potential Impact:*
    - Minimized unexpected downtime.
    - Reduced maintenance costs.
    - Improved overall operational efficiency and safety.
    """)

# ------------------------------
# 8. MAINTENANCE LOG
# ------------------------------
with tabs[7]:
    st.subheader("Maintenance Log")
    log = pd.DataFrame({
        'Date':['2025-11-01','2025-11-03','2025-11-05'],
        'Machine':['Motor #2','Pump #1','Bearing #4'],
        'Alert':['⚠ Warning','🔴 Critical','🟢 Normal'],
        'Action':['Schedule inspection','Immediate shutdown','No action required']
    })
    st.dataframe(log)
    st.download_button("Download Log", log.to_csv(index=False).encode('utf-8'), "maintenance_log.csv")

# ------------------------------
# 9. MODEL PERFORMANCE
# ------------------------------
with tabs[8]:
    st.subheader("Model Evaluation Metrics")
    if 'predicted_status' in data.columns:
        y_true = st.session_state.label_encoder.transform(data['status'])
        y_pred = st.session_state.label_encoder.transform(data['predicted_status'])

        st.text(classification_report(y_true, y_pred, target_names=st.session_state.label_encoder.classes_))

        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"),
                        x=st.session_state.label_encoder.classes_,
                        y=st.session_state.label_encoder.classes_,
                        color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

        # Binary ROC curve
        if len(st.session_state.label_encoder.classes_) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Baseline', line=dict(dash='dash')))
            fig_roc.update_layout(title=f"ROC Curve (AUC={roc_auc:.2f})", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig_roc, use_container_width=True)

st.sidebar.info("Developed by The Vanguards | Streamlit Predictive Maintenance Project")
