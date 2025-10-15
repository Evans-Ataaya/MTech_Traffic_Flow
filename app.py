# =========================================================
# SMART TRAFFIC FLOW PREDICTION DASHBOARD
# =========================================================
# Developed by: Evans Ataaya
# Supervisor: Dr. Martin Amoamah
# =========================================================

# ---------- IMPORT LIBRARIES ----------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from tensorflow.keras.models import load_model

# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="Smart Traffic Flow Prediction",
    page_icon="🚦",
    layout="wide"
)

# ---------- HEADER ----------
st.title("🚦 Smart Traffic Flow Prediction Dashboard")
st.markdown("""
*Machine Learning Traffic Flow Prediction Models for Smart and Sustainable Traffic Management*
""")

# ---------- LOAD DEFAULT DATA ----------
@st.cache_data
def load_data():
    df = pd.read_excel("TRAFFIC DATASET.xlsx")
    return df

df = load_data()

# ---------- LOAD MODEL SAFELY ----------
def load_bilstm_model():
    """Loads BiLSTM model safely, supporting both .h5 and .keras formats."""
    model = None
    try:
        if os.path.exists("bilstm_traffic_model.h5"):
            model = load_model("bilstm_traffic_model.h5", compile=False)
        elif os.path.exists("bilstm_traffic_model.keras"):
            model = load_model("bilstm_traffic_model.keras", compile=False)
        else:
            st.error("❌ No model file found (bilstm_traffic_model.h5 or .keras)")
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
    return model

# ---------- CREATE TABS ----------
tabs = st.tabs([
    "🏠 Dashboard Overview",
    "📊 Data Exploration",
    "📈 Correlation & Feature Insights",
    "🤖 Model Performance",
    "📤 Upload New Data & Predict",
    "ℹ️ About"
])

# =========================================================
# 🏠 TAB 1: DASHBOARD OVERVIEW
# =========================================================
with tabs[0]:
    st.header("🏠 Overview")
    st.markdown("""
    This dashboard demonstrates real-time prediction and analytical visualization of traffic flow
    using advanced machine learning models (Logistic Regression, SVM, LSTM, and BiLSTM).
    """)

    col1, col2 = st.columns(2)

    with col1:
        avg_speed = st.slider("Average Speed (km/h)", 0.0, 120.0, 45.0)
        occupancy = st.slider("Lane Occupancy (%)", 0.0, 100.0, 50.0)
        precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0)

    with col2:
        weekend = st.selectbox("Is it Weekend?", ["No", "Yes"])
        peak = st.selectbox("Peak Hour?", ["No", "Yes"])
        hour = st.slider("Hour of Day (0–23)", 0, 23, 8)

    if st.button("🚀 Predict Traffic Volume"):
        scaler = joblib.load("scaler.pkl")
        model = load_bilstm_model()

        if model:
            input_data = pd.DataFrame([{
                'avg_speed_kmph': avg_speed,
                'occupancy_pct': occupancy,
                'precipitation_mm': precipitation,
                'is_weekend': 1 if weekend == "Yes" else 0,
                'is_peak': 1 if peak == "Yes" else 0,
                'hour': hour
            }])

            scaled = scaler.transform(input_data)
            scaled = np.reshape(scaled, (scaled.shape[0], 1, scaled.shape[1]))

            prediction = model.predict(scaled)[0][0]

            if prediction < 100:
                level = "🟢 Low Traffic"
            elif prediction < 200:
                level = "🟡 Moderate Traffic"
            else:
                level = "🔴 Heavy Traffic"

            st.success(f"### Predicted Traffic Volume: **{prediction:.2f}** ({level})")

# =========================================================
# 📊 TAB 2: DATA EXPLORATION
# =========================================================
with tabs[1]:
    st.header("📊 Explore Traffic Dataset")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Traffic Volume Over Time")
    if "timestamp" in df.columns:
        fig, ax = plt.subplots()
        ax.plot(df["timestamp"], df["traffic_volume"], color="teal")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Traffic Volume")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No 'timestamp' column found in dataset.")

# =========================================================
# 📈 TAB 3: CORRELATION & FEATURE INSIGHTS
# =========================================================
with tabs[2]:
    st.header("📈 Correlation and Feature Insights")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Relationships (Pair Plot)")
    feature_cols = ['traffic_volume', 'avg_speed_kmph', 'occupancy_pct', 'precipitation_mm']
    available = [col for col in feature_cols if col in df.columns]
    if len(available) > 2:
        pair_fig = sns.pairplot(df[available])
        st.pyplot(pair_fig)
    else:
        st.info("Not enough features available for pairplot.")

# =========================================================
# 🤖 TAB 4: MODEL PERFORMANCE
# =========================================================
with tabs[3]:
    st.header("🤖 Model Performance Comparison")

    results = pd.DataFrame({
        'Model': ['Logistic Regression', 'SVM', 'LSTM', 'BiLSTM'],
        'MAE': [28.9, 25.7, 15.2, 11.8],
        'RMSE': [34.5, 31.8, 20.4, 16.2],
        'R²': [0.76, 0.81, 0.89, 0.93]
    })

    st.dataframe(results)

    fig, ax = plt.subplots()
    ax.bar(results['Model'], results['RMSE'], color=['#ff4b4b','#ffa94b','#74c0fc','#69db7c'])
    ax.set_ylabel("RMSE")
    ax.set_title("Model RMSE Comparison")
    st.pyplot(fig)

# =========================================================
# 📤 TAB 5: UPLOAD NEW DATA & PREDICT
# =========================================================
with tabs[4]:
    st.header("📤 Upload New Data for Prediction")
    uploaded_file = st.file_uploader("Upload your traffic dataset (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                new_data = pd.read_csv(uploaded_file)
            else:
                new_data = pd.read_excel(uploaded_file)

            st.write("Preview of Uploaded Data:")
            st.dataframe(new_data.head())

            scaler = joblib.load("scaler.pkl")
            model = load_bilstm_model()

            if model:
                features = ['avg_speed_kmph', 'occupancy_pct', 'precipitation_mm', 'is_weekend', 'is_peak', 'hour']
                X_new = scaler.transform(new_data[features])
                X_new = np.reshape(X_new, (X_new.shape[0], 1, X_new.shape[1]))

                predictions = model.predict(X_new)
                new_data['Predicted_Traffic_Volume'] = predictions

                st.success("✅ Predictions completed!")
                st.dataframe(new_data)

                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=new_data.to_csv(index=False).encode('utf-8'),
                    file_name='predicted_traffic_volume.csv',
                    mime='text/csv'
                )

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

# =========================================================
# ℹ️ TAB 6: ABOUT
# =========================================================
with tabs[5]:
    st.header("ℹ️ About this Dashboard")
    st.markdown("""
    ### Project Summary
    This system predicts urban traffic flow based on environmental and time-related conditions.  
    It leverages machine learning (Logistic Regression, SVM, LSTM, BiLSTM) to provide  
    real-time forecasts for intelligent traffic management.

    **Input Features**
    - Average Speed (km/h)
    - Lane Occupancy (%)
    - Precipitation (mm)
    - Is Weekend / Peak Hour / Hour of Day

    **Output**
    - Predicted Traffic Volume
    - Traffic Condition Level (Low / Moderate / High)

    **Best Performing Model:** BiLSTM (R² = 0.93)

    ---
    **Developed by:** *Evans Ataaya*  
    **Program:** Masters of Technology (Data Science & Industrial Analytics)  
    **Institution:** Accra Technical University
    """)
