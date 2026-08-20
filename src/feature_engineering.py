"""
Feature Engineering Module

This module creates all features needed for the XGBoost model:
- Time-based features (hour, day, month, quarter)
- Cyclical encoding (sin/cos transformation)
- Lag features (1-48 hours)
- Rolling statistics (24-hour mean and standard deviation)
- Weather features (temperature, humidity, wind speed)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Get the project root directory
BASE_DIR = Path(__file__).parent.parent.absolute()


def create_time_features(df):
    """
    Create time-based features from datetime index.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added time features
    """
    data = df.copy()
    
    # Ensure datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'Datetime' in data.columns:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data = data.set_index('Datetime')
    
    # Extract time components
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter
    data['year'] = data.index.year
    
    return data


def create_cyclical_features(df):
    """
    Create cyclical (sin/cos) encoding for time features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with hour and day_of_week columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added cyclical features
    """
    data = df.copy()
    
    # Hour of day (0-23) -> sin/cos
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    
    # Day of week (0-6) -> sin/cos
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    return data


def create_lag_features(df, target_col='PJME_MW', lags=[1, 2, 3, 6, 12, 24, 48]):
    """
    Create lag features for the target variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with target column
    target_col : str
        Name of the target column
    lags : list
        List of lag values to create
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added lag features
    """
    data = df.copy()
    
    for lag in lags:
        data[f'lag_{lag}h'] = data[target_col].shift(lag)
    
    return data


def create_rolling_features(df, target_col='PJME_MW', window=24):
    """
    Create rolling statistics features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with target column
    target_col : str
        Name of the target column
    window : int
        Rolling window size
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added rolling features
    """
    data = df.copy()
    
    data[f'rolling_mean_{window}h'] = data[target_col].rolling(window=window).mean()
    data[f'rolling_std_{window}h'] = data[target_col].rolling(window=window).std()
    
    return data


def engineer_features(df, target_col='PJME_MW'):
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index and target column
    target_col : str
        Name of the target column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all engineered features
    """
    print("Starting feature engineering...")
    
    # Step 1: Create time features
    print("  Creating time features...")
    data = create_time_features(df)
    
    # Step 2: Create cyclical features
    print("  Creating cyclical features...")
    data = create_cyclical_features(data)
    
    # Step 3: Create lag features
    print("  Creating lag features...")
    data = create_lag_features(data, target_col)
    
    # Step 4: Create rolling features
    print("  Creating rolling features...")
    data = create_rolling_features(data, target_col)
    
    # Step 5: Drop rows with NaN (from lags and rolling)
    print("  Dropping NaN rows...")
    data = data.dropna()
    
    print(f"  Feature engineering complete! Shape: {data.shape}")
    print(f"  Number of features: {len(data.columns) - 1}")
    
    return data


def get_feature_columns(df, exclude_col='PJME_MW'):
    """
    Get list of feature columns (exclude target).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features
    exclude_col : str
        Column to exclude (target)
        
    Returns:
    --------
    list
        List of feature column names
    """
    return [col for col in df.columns if col != exclude_col]


def prepare_features_and_target(df, target_col='PJME_MW'):
    """
    Prepare X (features) and y (target) for model training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features and target
    target_col : str
        Name of the target column
        
    Returns:
    --------
    tuple
        (X, y, feature_cols)
    """
    # Get feature columns
    feature_cols = get_feature_columns(df, target_col)
    
    # Split into X and y
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y, feature_cols


def engineer_weather_features(df):
    """
    Add weather features to the dataset if available.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with weather data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with weather features included
    """
    data = df.copy()
    
    weather_cols = ['Temperature', 'Humidity', 'WindSpeed', 'Pressure', 'Clouds']
    
    for col in weather_cols:
        if col in data.columns:
            print(f"  Added weather feature: {col}")
        else:
            # Create placeholder columns if weather data not available
            data[col] = np.nan
    
    return data


if __name__ == "__main__":
    # Quick test when run directly
    print("Testing feature engineering module...")
    
    # Load sample data
    try:
        df = pd.read_csv(BASE_DIR / 'PJME_hourly.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.set_index('Datetime')
        
        print(f"Original data shape: {df.shape}")
        
        # Run feature engineering
        df_engineered = engineer_features(df)
        
        print(f"Engineered data shape: {df_engineered.shape}")
        print(f"Features: {list(df_engineered.columns)}")
        print("\n✅ Feature engineering module working correctly!")
        
    except FileNotFoundError:
        print("⚠️ PJME_hourly.csv not found. Skipping test.")
