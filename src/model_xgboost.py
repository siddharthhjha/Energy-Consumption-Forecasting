"""
XGBoost Model Training Module

This module handles training, tuning, and saving of the XGBoost model
for energy consumption forecasting.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import joblib
from pathlib import Path

# Get the project root directory
BASE_DIR = Path(__file__).parent.parent.absolute()

def create_features(df):
    """
    Create features for XGBoost model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index and PJME_MW column
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    feature_cols : list
        List of feature column names
    """
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Ensure datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'Datetime' in data.columns:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data = data.set_index('Datetime')
    
    # Time features
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter
    data['year'] = data.index.year
    
    # Cyclical encoding
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        data[f'lag_{lag}h'] = data['PJME_MW'].shift(lag)
    
    # Rolling statistics
    data['rolling_mean_24h'] = data['PJME_MW'].rolling(window=24).mean()
    data['rolling_std_24h'] = data['PJME_MW'].rolling(window=24).std()
    
    # Drop rows with NaN (from lags and rolling features)
    data = data.dropna()
    
    # Define feature columns
    feature_cols = [
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'month', 'quarter',
        'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h',
        'rolling_mean_24h', 'rolling_std_24h'
    ]
    
    # Add weather features if they exist
    weather_cols = ['Temperature', 'Humidity', 'WindSpeed', 'Pressure', 'Clouds']
    for col in weather_cols:
        if col in data.columns:
            feature_cols.append(col)
    
    X = data[feature_cols]
    y = data['PJME_MW']
    
    return X, y, feature_cols

def train_xgboost(df, test_size=0.2, random_state=42):
    """
    Train XGBoost model on the given data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features and target
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    model : xgboost.XGBRegressor
        Trained XGBoost model
    metrics : dict
        Dictionary of evaluation metrics
    predictions : pandas.DataFrame
        DataFrame with actual and predicted values
    """
    # Create features
    X, y, feature_cols = create_features(df)
    
    # Split data chronologically (important for time series)
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    metrics = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2)
    }
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Create predictions DataFrame
    predictions = pd.DataFrame({
        'Datetime': X_test.index,
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    
    # Save model
    model_path = BASE_DIR / 'models' / 'xgboost_energy_model.json'
    model_path.parent.mkdir(exist_ok=True)
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = BASE_DIR / 'outputs' / 'feature_importance.csv'
    importance_path.parent.mkdir(exist_ok=True)
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")
    
    # Save predictions
    predictions_path = BASE_DIR / 'outputs' / 'predictions.csv'
    predictions.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    metrics_path = BASE_DIR / 'outputs' / 'model_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")
    
    return model, metrics, predictions

def load_model(model_path=None):
    """
    Load a trained XGBoost model.
    
    Parameters:
    -----------
    model_path : str or Path
        Path to the saved model file
        
    Returns:
    --------
    model : xgboost.XGBRegressor
        Loaded XGBoost model
    """
    if model_path is None:
        model_path = BASE_DIR / 'models' / 'xgboost_energy_model.json'
    
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    return model

def predict_future(model, last_data, hours_ahead=24):
    """
    Make future predictions using the trained model.
    
    Parameters:
    -----------
    model : xgboost.XGBRegressor
        Trained XGBoost model
    last_data : pandas.DataFrame
        Last known data point(s) with all features
    hours_ahead : int
        Number of hours to forecast ahead
        
    Returns:
    --------
    predictions : pandas.DataFrame
        Future predictions with timestamps
    """
    # This is a placeholder for future prediction logic
    # In a real implementation, you would roll the predictions forward
    print("Future prediction functionality coming soon...")
    return None

if __name__ == "__main__":
    # Quick test when run directly
    print("Loading data...")
    df = pd.read_csv(BASE_DIR / 'PJME_hourly.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    
    # Train model
    model, metrics, predictions = train_xgboost(df)
    print("\n✅ Model training complete!")
