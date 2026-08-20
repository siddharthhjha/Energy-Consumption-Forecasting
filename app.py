# app.py - Streamlit Dashboard for Energy Forecasting
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Consumption Forecasting Dashboard")
st.markdown("### PJM East Region - 24-Hour Electricity Demand Forecasting")

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    """Load processed data and model outputs."""
    try:
        df = pd.read_csv('PJME_hourly.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df
    except FileNotFoundError:
        st.error("PJME_hourly.csv not found. Please check the file path.")
        return None

@st.cache_data
def load_outputs():
    """Load model outputs from outputs folder."""
    outputs = {}
    try:
        outputs['metrics'] = pd.read_csv('outputs/model_metrics.csv')
        outputs['importance'] = pd.read_csv('outputs/feature_importance.csv')
        outputs['predictions'] = pd.read_csv('outputs/predictions.csv')
        outputs['comparison'] = pd.read_csv('outputs/model_comparison.csv')
        return outputs
    except FileNotFoundError as e:
        st.warning(f"Some output files not found: {e}")
        return None

# Load everything
df = load_data()
outputs = load_outputs()

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "Data & EDA", "Weather", "Features", "Model & Evaluation"]
)

# ============================================
# PAGE 1: OVERVIEW
# ============================================
if page == "Overview":
    st.header("📊 Project Overview")
    
    if outputs and df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        # Get metrics
        metrics_dict = {row['Metric']: row['Value'] for _, row in outputs['metrics'].iterrows()}
        
        col1.metric("MAPE", f"{metrics_dict.get('MAPE', 0):.2f}%")
        col2.metric("R² Score", f"{metrics_dict.get('R2', 0):.4f}")
        col3.metric("RMSE", f"{metrics_dict.get('RMSE', 0):.2f} MW")
        col4.metric("Records", f"{len(df):,}")
    
    st.markdown("""
    ### About This Project
    
    This project estimates PJM East's electricity demand one day ahead to support:
    - Grid balancing
    - Operational planning
    - Cost control
    
    **Features used:**
    - Historical load (PJME_MW)
    - Calendar features (hour, day, month)
    - Lag features (1h, 2h, 3h, 6h, 12h, 24h, 48h)
    - Rolling statistics (24h mean/std)
    - Weather variables (temperature, humidity, wind speed)
    
    **Model:** XGBoost Regressor
    """)
    
    # Show predictions plot if available
    if os.path.exists('outputs/predictions_plot.png'):
        st.image('outputs/predictions_plot.png', caption='Actual vs Predicted', use_container_width=True)

# ============================================
# PAGE 2: DATA & EDA
# ============================================
elif page == "Data & EDA":
    st.header("📈 Data & Exploratory Analysis")
    
    if df is not None:
        # Show data summary
        st.subheader("Data Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Records:** {len(df):,}")
            st.write(f"**Date Range:** {df['Datetime'].min()} to {df['Datetime'].max()}")
        with col2:
            st.write("**Columns:**")
            st.write(df.columns.tolist())
        
        # Show raw data
        with st.expander("View Raw Data"):
            st.dataframe(df.head(100))
        
        # Plot time series
        st.subheader("Energy Consumption Over Time")
        fig = px.line(df, x='Datetime', y='PJME_MW', title='Hourly Energy Consumption')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show EDA plot
        if os.path.exists('outputs/eda_overview.png'):
            st.image('outputs/eda_overview.png', caption='EDA Overview', use_container_width=True)

# ============================================
# PAGE 3: WEATHER
# ============================================
elif page == "Weather":
    st.header("🌤️ Weather Data Integration")
    
    st.markdown("""
    Weather data is pulled from **OpenWeatherMap API** using:
    - Latitude: 39.95°N
    - Longitude: 75.17°W (PJM East region)
    
    **Weather features:**
    - Temperature (°C)
    - Humidity (%)
    - Wind Speed (m/s)
    - Pressure (hPa)
    - Cloud Cover (%)
    """)
    
    try:
        weather_df = pd.read_csv('data/processed/weather_data.csv')
        st.subheader("Weather Data Sample")
        st.dataframe(weather_df.head(50))
        
        # Temperature plot
        fig = px.line(weather_df, x='Datetime', y='Temperature', title='Temperature Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Weather data not found. Run the data pipeline first.")

# ============================================
# PAGE 4: FEATURES
# ============================================
elif page == "Features":
    st.header("🧩 Feature Engineering")
    
    if outputs is not None:
        # Show feature importance
        st.subheader("Top 10 Most Important Features")
        fig = px.bar(outputs['importance'].head(10), 
                     x='importance', y='feature', 
                     orientation='h',
                     title='Feature Importance',
                     color='importance',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Features engineered:**
        - **Time Features:** hour, day_of_week, month, quarter
        - **Cyclical Encoding:** hour_sin, hour_cos, day_sin, day_cos
        - **Lag Features:** lag_1h, lag_2h, lag_3h, lag_6h, lag_12h, lag_24h, lag_48h
        - **Rolling Statistics:** rolling_mean_24h, rolling_std_24h
        - **Weather Features:** Temperature, Humidity, WindSpeed
        """)
    
    if os.path.exists('outputs/feature_importance.png'):
        st.image('outputs/feature_importance.png', caption='Feature Importance', use_container_width=True)

# ============================================
# PAGE 5: MODEL & EVALUATION
# ============================================
else:
    st.header("🤖 Model & Evaluation")
    
    if outputs is not None:
        st.subheader("Model Performance Metrics")
        st.dataframe(outputs['metrics'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Comparison")
            st.dataframe(outputs['comparison'])
        
        with col2:
            st.subheader("Predictions Sample")
            st.dataframe(outputs['predictions'].head(20))
        
        # Show residuals plot
        if os.path.exists('outputs/residuals_plot.png'):
            st.image('outputs/residuals_plot.png', caption='Residuals Analysis', use_container_width=True)
    
    st.markdown("""
    ### Model Details
    
    **Algorithm:** XGBoost Regressor
    
    **Hyperparameters:**
    - n_estimators: 1000
    - learning_rate: 0.01
    - max_depth: 6
    - subsample: 0.8
    - colsample_bytree: 0.8
    
    **Architecture:**
    At forecast origin t, the model predicts demand at t+24 hours.
    Training/test records are split by target timestamp, so evaluation
    does not use future observed load as an input.
    """)

# ============================================
# FOOTER
# ============================================
st.sidebar.markdown("---")
st.sidebar.info("""
**Project Info**
- Model: XGBoost
- Region: PJM East
- Horizon: 24 hours
- MAPE: <5%
""")

st.markdown("---")
st.markdown("Built with Streamlit • Data: PJM Hourly Energy Consumption • Weather: OpenWeatherMap API")
