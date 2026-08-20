# app.py - Streamlit Dashboard for Energy Forecasting
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent.absolute()

# Add src to path
sys.path.append(str(BASE_DIR / 'src'))

st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Consumption Forecasting Dashboard")
st.markdown("### PJM East Region - 24-Hour Electricity Demand Forecasting")

# ============================================
# HELPER FUNCTION TO FIND FILES
# ============================================
def find_file(filename, search_paths):
    """Search for a file in multiple paths."""
    for path in search_paths:
        full_path = Path(path) / filename
        if full_path.exists():
            return full_path
    return None

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    """Load raw PJM data."""
    # Try multiple possible locations
    possible_paths = [
        BASE_DIR / 'PJME_hourly.csv',
        BASE_DIR / 'data' / 'PJME_hourly.csv',
        BASE_DIR / '..' / 'PJME_hourly.csv'
    ]
    
    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path)
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            return df
    
    st.warning("PJME_hourly.csv not found. Please check the file path.")
    return None

@st.cache_data
def load_outputs():
    """Load all model outputs with fallback to generated data."""
    outputs = {
        'metrics': None,
        'importance': None,
        'predictions': None,
        'comparison': None
    }
    
    # Try multiple locations for outputs
    output_dirs = [
        BASE_DIR / 'outputs',
        BASE_DIR / '..' / 'outputs',
        BASE_DIR / 'data' / 'outputs'
    ]
    
    for out_dir in output_dirs:
        if out_dir.exists():
            # Check each file
            for file_name, key in [
                ('model_metrics.csv', 'metrics'),
                ('feature_importance.csv', 'importance'),
                ('predictions.csv', 'predictions'),
                ('model_comparison.csv', 'comparison')
            ]:
                file_path = out_dir / file_name
                if file_path.exists() and outputs[key] is None:
                    outputs[key] = pd.read_csv(file_path)
            
            # If we found metrics, break
            if outputs['metrics'] is not None:
                break
    
    return outputs

@st.cache_data
def load_weather():
    """Load weather data from multiple possible locations."""
    possible_paths = [
        BASE_DIR / 'data' / 'processed' / 'weather_data.csv',
        BASE_DIR / 'weather_data.csv',
        BASE_DIR / '..' / 'data' / 'processed' / 'weather_data.csv'
    ]
    
    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)
    
    return None

# Load everything with proper error handling
df = load_data()
outputs = load_outputs()
weather_df = load_weather()

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
    
    if outputs and outputs['metrics'] is not None:
        col1, col2, col3, col4 = st.columns(4)
        metrics_dict = {row['Metric']: row['Value'] for _, row in outputs['metrics'].iterrows()}
        
        col1.metric("MAPE", f"{metrics_dict.get('MAPE', 0):.2f}%")
        col2.metric("R² Score", f"{metrics_dict.get('R2', 0):.4f}")
        col3.metric("RMSE", f"{metrics_dict.get('RMSE', 0):.2f} MW")
        col4.metric("Records", f"{len(df):,}" if df is not None else "N/A")
    else:
        st.info("Run the model pipeline to generate metrics. Click the button below to train the model.")
        if st.button("🚀 Run Model Pipeline"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Import and run the pipeline
                    from src.model_xgboost import train_xgboost
                    from src.evaluation import evaluate_model
                    
                    if df is not None:
                        model, metrics, predictions = train_xgboost(df)
                        st.success("Model trained successfully! Refresh the page to see results.")
                    else:
                        st.error("Data not available. Please upload PJME_hourly.csv")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Make sure you have all dependencies installed and run the pipeline locally first.")
    
    st.markdown("""
    ### About This Project
    
    This project estimates PJM East's electricity demand one day ahead to support:
    - Grid balancing
    - Operational planning
    - Cost control
    """)
    
    # Show predictions plot if available
    pred_plot = find_file('predictions_plot.png', [BASE_DIR / 'outputs'])
    if pred_plot and pred_plot.exists():
        st.image(str(pred_plot), caption='Actual vs Predicted', use_container_width=True)

# ============================================
# PAGE 2: DATA & EDA
# ============================================
elif page == "Data & EDA":
    st.header("📈 Data & Exploratory Analysis")
    
    if df is not None:
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
        sample_df = df.sample(min(5000, len(df)))
        fig = px.line(sample_df, x='Datetime', y='PJME_MW', title='Hourly Energy Consumption')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data not loaded. Please check if PJME_hourly.csv is in the repository.")

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
    
    if weather_df is not None:
        st.subheader("Weather Data Sample")
        st.dataframe(weather_df.head(50))
        
        if 'Temperature' in weather_df.columns and 'Datetime' in weather_df.columns:
            fig = px.line(weather_df, x='Datetime', y='Temperature', title='Temperature Over Time')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Weather data not found. Run the data pipeline first.")
        st.info("""
        To generate weather data, run:
        ```bash
        python -m src.data_pipeline
