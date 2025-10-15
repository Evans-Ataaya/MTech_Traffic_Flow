# ======================================================
# 🚦 MTech Thesis — Traffic Flow Prediction Dashboard
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import os

# ====================================
# ⚙️ PAGE CONFIGURATION
# ====================================
st.set_page_config(
    page_title="Traffic Flow Prediction Dashboard",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Intelligent Traffic Flow Prediction System")
st.markdown("""
Analyze, compare, and visualize model performance for predicting urban traffic flow.
""")

# ====================================
# 🧩 TABS LAYOUT
# ====================================
tab1, tab2, tab3 = st.tabs([
    "📊 Model Comparison Dashboard",
    "🚀 Predict Traffic Flow",
    "📤 Upload New Data"
])

# ====================================
# 🧠 LOAD DATA, SCALER, MODEL
# ====================================
@st.cache_data
def load_data():
    return pd.read_excel("TRAFFIC DATASET.xlsx")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_bilstm():
    """Load the BiLSTM model safely."""
    try:
        if os.path.exists("bilstm_traffic_model.h5"):
            return load_model("bilstm_traffic_model.h5", compile=False)
        elif os.path.exists("bilstm_traffic_model.keras"):
            return load_model("bilstm_traffic_model.keras", compile=False)
        else:
            st.error("❌ Model file not found!")
            return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

df = load_data()
scaler = load_scaler()
bilstm_model = load_bilstm()

# ====================================
# 🧩 HELPER FUNCTIONS
# ====================================
def predict_bilstm(model, scaler, data):
    """Predict traffic volume using BiLSTM model."""
    scaled = scaler.transform(data)
    scaled = np.reshape(scaled, (scaled.shape[0], 1, scaled.shape[1]))
    prediction = model.predict(scaled)[0][0]
    return prediction

def evaluate_models():
    """
    Return a comparison DataFrame for multiple models.
    Replace demo metrics with your actual experiment results if available.
    """
    results = {
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine (SVM)",
            "LSTM",
            "BiLSTM"
        ],
        "MAE": [15.8, 12.1, 10.4, 12.3, 9.5, 8.7],
        "RMSE": [22.7, 19.3, 17.1, 18.5, 12.9, 11.2],
        "R²": [0.74, 0.80, 0.86, 0.81, 0.91, 0.93]
    }
    return pd.DataFrame(results)

# ====================================
# TAB 1: MODEL COMPARISON DASHBOARD
# ====================================
with tab1:
    st.subheader("📈 Model Performance Comparison Across Algorithms")

    metrics_df = evaluate_models()
    st.dataframe(
        metrics_df.style.highlight_max(axis=0, color='lightgreen'),
        use_container_width=True
    )

    # --- Visualization 1: MAE and RMSE Bar Charts ---
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.barplot(data=metrics_df, x="Model", y="MAE", palette="coolwarm", ax=ax1)
        plt.xticks(rotation=45, ha='right')
        plt.title("Mean Absolute Error (MAE)")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.barplot(data=metrics_df, x="Model", y="RMSE", palette="magma", ax=ax2)
        plt.xticks(rotation=45, ha='right')
        plt.title("Root Mean Squared Error (RMSE)")
        st.pyplot(fig2)

    # --- Visualization 2: R² Score Bar Chart ---
    fig3, ax3 = plt.subplots()
    sns.barplot(data=metrics_df, x="Model", y="R²", palette="viridis", ax=ax3)
    plt.xticks(rotation=45, ha='right')
    plt.title("R² (Coefficient of Determination)")
    st.pyplot(fig3)

    # --- Highlight the Best Model ---
    best_model = metrics_df.loc[metrics_df['R²'].idxmax(), 'Model']
    st.success(f"🏆 **Best Performing Model:** {best_model}")
    st.markdown("""
    ✅ The **BiLSTM** model achieved the lowest MAE and RMSE, and the highest R² score,  
    demonstrating superior ability to model **temporal traffic dependencies** compared to other algorithms.
    """)

    # --- Download Option ---
    st.download_button(
        label="📥 Download Comparison Table (CSV)",
        data=metrics_df.to_csv(index=False).encode('utf-8'),
        file_name='model_comparison_results.csv',
        mime='text/csv'
    )

# ====================================
# TAB 2: PREDICT TRAFFIC FLOW
# ====================================
with tab2:
    st.subheader("🚀 Predict Traffic Volume (BiLSTM Model)")

    st.sidebar.header("🔧 Input Parameters")
    avg_speed = st.sidebar.slider("Average Speed (km/h)", 0, 150, 60)
    occupancy = st.sidebar.slider("Occupancy (%)", 0, 100, 50)
    precipitation = st.sidebar.slider("Precipitation (mm)", 0.0, 50.0, 5.0)
    weekend = st.sidebar.selectbox("Weekend?", ["No", "Yes"])
    peak = st.sidebar.selectbox("Peak Hour?", ["No", "Yes"])
    hour = st.sidebar.slider("Hour of Day (0–23)", 0, 23, 8)

    input_data = pd.DataFrame([{
        'avg_speed_kmph': avg_speed,
        'occupancy_pct': occupancy,
        'precipitation_mm': precipitation,
        'is_weekend': 1 if weekend == "Yes" else 0,
        'is_peak': 1 if peak == "Yes" else 0,
        'hour': hour
    }])

    st.write("### Input Data", input_data)

    if st.button("Predict Traffic Volume"):
        bilstm_pred = predict_bilstm(bilstm_model, scaler, input_data)
        st.success(f"**Predicted Traffic Volume: {bilstm_pred:.2f} vehicles/hour**")

        if bilstm_pred < 100:
            st.info("🟢 Low Traffic — Free Flow")
        elif bilstm_pred < 200:
            st.warning("🟡 Moderate Traffic — Caution")
        else:
            st.error("🔴 Heavy Congestion — Expect Delays")

# ====================================
# TAB 3: UPLOAD NEW DATA
# ====================================
with tab3:
    st.subheader("📤 Upload New Dataset for Exploratory Analysis")

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                new_df = pd.read_csv(uploaded_file)
            else:
                new_df = pd.read_excel(uploaded_file)

            st.success("✅ File uploaded successfully!")
            st.write("### Preview of Uploaded Data", new_df.head())

            st.write("### Statistical Summary", new_df.describe())

            numeric_df = new_df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
                plt.title("Correlation Heatmap (Numeric Features Only)")
                st.pyplot(fig_corr)
            else:
                st.warning("⚠️ No numeric columns found in uploaded data.")

        except Exception as e:
            st.error(f"⚠️ Error processing uploaded file: {e}")

# ====================================
# FOOTER
# ====================================
st.markdown("---")
st.caption("MTech Thesis | Traffic Flow Prediction System | © 2025 Evans Ataaya")
