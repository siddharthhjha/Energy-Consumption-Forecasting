import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import traceback
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
# SESSION STATE FOR TRAINING STATUS
# ============================================
if 'training' not in st.session_state:
    st.session_state.training = False
if 'training_done' not in st.session_state:
    st.session_state.training_done = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

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
    
    return None

@st.cache_data
def load_outputs():
    """Load all model outputs."""
    outputs = {
        'metrics': None,
        'importance': None,
        'predictions': None,
        'comparison': None
    }
    
    output_dirs = [
        BASE_DIR / 'outputs',
        BASE_DIR / '..' / 'outputs',
        BASE_DIR / 'data' / 'outputs'
    ]
    
    for out_dir in output_dirs:
        if out_dir.exists():
            for file_name, key in [
                ('model_metrics.csv', 'metrics'),
                ('feature_importance.csv', 'importance'),
                ('predictions.csv', 'predictions'),
                ('model_comparison.csv', 'comparison')
            ]:
                file_path = out_dir / file_name
                if file_path.exists() and outputs[key] is None:
                    outputs[key] = pd.read_csv(file_path)
            
            if outputs['metrics'] is not None:
                break
    
    return outputs

def train_model_pipeline(df):
    """Run the model training pipeline with detailed error reporting."""
    try:
        st.text("Step 1: Importing required modules...")
        from src.feature_engineering import engineer_features
        st.text("✓ Feature engineering module imported")
        
        from src.model_xgboost import train_xgboost
        st.text("✓ XGBoost module imported")
        
        from src.evaluation import evaluate_model
        st.text("✓ Evaluation module imported")
        
        st.text("Step 2: Engineering features...")
        df_engineered = engineer_features(df)
        st.text(f"✓ Features engineered: {len(df_engineered.columns)} columns")
        
        st.text("Step 3: Training XGBoost model...")
        model, metrics, predictions = train_xgboost(df_engineered)
        st.text("✓ Model training complete!")
        
        st.text("Step 4: Evaluating model...")
        evaluate_model(model, metrics, predictions)
        st.text("✓ Evaluation complete!")
        
        st.session_state.training_done = True
        st.session_state.training = False
        st.success("✅ Model training completed successfully!")
        
    except Exception as e:
        st.session_state.training = False
        st.session_state.error_message = str(e)
        st.error(f"❌ Error during training: {str(e)}")
        st.code(traceback.format_exc())

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
    st.header("Project Overview")
    
    # Load data
    df = load_data()
    outputs = load_outputs()
    
    if df is None:
        st.error("❌ PJME_hourly.csv not found. Please check the file path.")
        st.info(f"Looking for file in: {BASE_DIR}")
        if BASE_DIR.exists():
            st.write("Files in directory:", list(BASE_DIR.glob("*.csv")))
    
    if outputs and outputs['metrics'] is not None:
        col1, col2, col3, col4 = st.columns(4)
        metrics_dict = {row['Metric']: row['Value'] for _, row in outputs['metrics'].iterrows()}
        
        col1.metric("MAPE", f"{metrics_dict.get('MAPE', 0):.2f}%")
        col2.metric("R² Score", f"{metrics_dict.get('R2', 0):.4f}")
        col3.metric("RMSE", f"{metrics_dict.get('RMSE', 0):.2f} MW")
        col4.metric("Records", f"{len(df):,}" if df is not None else "N/A")
        
        st.success("✅ Model metrics loaded successfully!")
        
        # Show predictions plot
        pred_plot = find_file('predictions_plot.png', [BASE_DIR / 'outputs'])
        if pred_plot and pred_plot.exists():
            st.image(str(pred_plot), caption='Actual vs Predicted', use_container_width=True)
    
    elif df is not None:
        st.info("Run the model pipeline to generate metrics.")
        
        if st.button("Run Model Pipeline"):
            st.session_state.training = True
            st.session_state.training_done = False
            st.session_state.error_message = None
            
            with st.spinner("Training model... This may take a few minutes."):
                train_model_pipeline(df)
        
        # Show training status
        if st.session_state.training:
            st.warning("⏳ Training in progress... Please wait.")
        
        if st.session_state.error_message:
            st.error(f"⚠️ Error: {st.session_state.error_message}")
            st.info("Check the logs for more details.")
    
    st.markdown("""
    ### About This Project
    
    This project estimates PJM East's electricity demand one day ahead to support:
    - Grid balancing
    - Operational planning
    - Cost control
    """)
    
    st.markdown("---")
    st.caption("Built with Streamlit • Data: PJM Hourly Energy Consumption • Weather: OpenWeatherMap API")

# ============================================
# PAGE 2: DATA & EDA
# ============================================
elif page == "Data & EDA":
    st.header("Data & Exploratory Analysis")
    
    df = load_data()
    
    if df is not None:
        st.subheader("Data Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Records:** {len(df):,}")
            st.write(f"**Date Range:** {df['Datetime'].min()} to {df['Datetime'].max()}")
        with col2:
            st.write("**Columns:**")
            st.write(df.columns.tolist())
        
        with st.expander("View Raw Data"):
            st.dataframe(df.head(100))
        
        st.subheader("Energy Consumption Over Time")
        import plotly.express as px
        sample_df = df.sample(min(5000, len(df)))
        fig = px.line(sample_df, x='Datetime', y='PJME_MW', title='Hourly Energy Consumption')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data not loaded. Please check if PJME_hourly.csv is in the repository.")

# ============================================
# PAGE 3: WEATHER
# ============================================
elif page == "Weather":
    st.header("Weather Data Integration")
    
    st.markdown("""
    Weather data is pulled from **OpenWeatherMap API** using:
    - Latitude: 39.95°N
    - Longitude: 75.17°W (PJM East region)
    
    **Weather features:**
    - Temperature (C)
    - Humidity (%)
    - Wind Speed (m/s)
    - Pressure (hPa)
    - Cloud Cover (%)
    """)
    
    weather_path = BASE_DIR / 'data' / 'processed' / 'weather_data.csv'
    if weather_path.exists():
        weather_df = pd.read_csv(weather_path)
        st.subheader("Weather Data Sample")
        st.dataframe(weather_df.head(50))
        
        if 'Temperature' in weather_df.columns and 'Datetime' in weather_df.columns:
            import plotly.express as px
            fig = px.line(weather_df, x='Datetime', y='Temperature', title='Temperature Over Time')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Weather data not found. Run the data pipeline first.")
        st.info(f"Looking for: {weather_path}")

# ============================================
# PAGE 4: FEATURES
# ============================================
elif page == "Features":
    st.header("Feature Engineering")
    
    outputs = load_outputs()
    
    if outputs and outputs['importance'] is not None:
        import plotly.express as px
        st.subheader("Top 10 Most Important Features")
        fig = px.bar(outputs['importance'].head(10), 
                     x='importance', y='feature', 
                     orientation='h',
                     title='Feature Importance',
                     color='importance',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance data not available. Run the model pipeline to generate it.")
    
    st.markdown("""
    **Features engineered:**
    - **Time Features:** hour, day_of_week, month, quarter
    - **Cyclical Encoding:** hour_sin, hour_cos, day_sin, day_cos
    - **Lag Features:** lag_1h, lag_2h, lag_3h, lag_6h, lag_12h, lag_24h, lag_48h
    - **Rolling Statistics:** rolling_mean_24h, rolling_std_24h
    - **Weather Features:** Temperature, Humidity, WindSpeed
    """)
    
    imp_plot = find_file('feature_importance.png', [BASE_DIR / 'outputs'])
    if imp_plot and imp_plot.exists():
        st.image(str(imp_plot), caption='Feature Importance', use_container_width=True)

# ============================================
# PAGE 5: MODEL & EVALUATION
# ============================================
else:
    st.header("Model & Evaluation")
    
    outputs = load_outputs()
    
    if outputs and outputs['metrics'] is not None:
        st.subheader("Model Performance Metrics")
        st.dataframe(outputs['metrics'])
        
        if outputs['comparison'] is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Model Comparison")
                st.dataframe(outputs['comparison'])
        
        if outputs['predictions'] is not None:
            st.subheader("Predictions Sample")
            st.dataframe(outputs['predictions'].head(20))
        
        res_plot = find_file('residuals_plot.png', [BASE_DIR / 'outputs'])
        if res_plot and res_plot.exists():
            st.image(str(res_plot), caption='Residuals Analysis', use_container_width=True)
    else:
        st.info("No model outputs found. Run the pipeline locally and commit the outputs/ folder.")
    
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
