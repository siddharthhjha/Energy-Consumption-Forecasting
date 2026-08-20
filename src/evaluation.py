"""
Evaluation Module

This module handles model evaluation, metrics calculation, and visualization.
It generates all performance metrics and plots for the energy forecasting project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import os

# Get the project root directory
BASE_DIR = Path(__file__).parent.parent.absolute()

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def calculate_metrics(y_true, y_pred):
    """
    Calculate all evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary containing all metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2)
    }
    
    return metrics


def plot_predictions(y_true, y_pred, title="Actual vs Predicted Energy Consumption", save_path=None):
    """
    Create actual vs predicted plot.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str or Path
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create time index for plotting
    time_idx = np.arange(len(y_true))
    
    # Plot actual and predicted
    ax.plot(time_idx, y_true, label='Actual', linewidth=2, alpha=0.7)
    ax.plot(time_idx, y_pred, label='Predicted', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Energy Consumption (MW)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to: {save_path}")
    
    return fig


def plot_residuals(y_true, y_pred, title="Residuals Analysis", save_path=None):
    """
    Create residuals analysis plot.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str or Path
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Residuals over time
    axes[0].scatter(range(len(residuals)), residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Time (hours)')
    axes[0].set_ylabel('Residuals (MW)')
    axes[0].set_title('Residuals Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals (MW)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residuals plot saved to: {save_path}")
    
    return fig


def plot_feature_importance(importance_df, top_n=10, title="Feature Importance", save_path=None):
    """
    Create feature importance plot.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to show
    title : str
        Plot title
    save_path : str or Path
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Sort and get top N
    top_features = importance_df.nlargest(top_n, 'importance')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Horizontal bar chart
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))
    ax.barh(top_features['feature'], top_features['importance'], color=colors)
    
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'{title} - Top {top_n} Features')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    return fig


def plot_scatter_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted Scatter", save_path=None):
    """
    Create scatter plot of actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str or Path
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual (MW)')
    ax.set_ylabel('Predicted (MW)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {save_path}")
    
    return fig


def save_metrics_to_csv(metrics, save_path=None):
    """
    Save metrics to CSV file.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    save_path : str or Path
        Path to save the CSV
    """
    if save_path is None:
        save_path = BASE_DIR / 'outputs' / 'model_metrics.csv'
    
    # Ensure directory exists
    save_path.parent.mkdir(exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    
    metrics_df.to_csv(save_path, index=False)
    print(f"Metrics saved to: {save_path}")
    
    return metrics_df


def print_metrics(metrics):
    """
    Print metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    """
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    
    for metric, value in metrics.items():
        if metric == 'R2':
            print(f"{metric}: {value:.4f}")
        elif metric == 'MAPE':
            print(f"{metric}: {value:.2f}%")
        else:
            print(f"{metric}: {value:.2f}")
    
    print("="*60 + "\n")


def evaluate_model(model, metrics, predictions, feature_cols=None):
    """
    Complete evaluation pipeline - generates all metrics and visualizations.
    
    Parameters:
    -----------
    model : XGBoost model
        Trained model (for feature importance)
    metrics : dict
        Dictionary of metrics
    predictions : pandas.DataFrame
        DataFrame with predictions
    feature_cols : list
        List of feature column names (optional)
        
    Returns:
    --------
    dict
        Dictionary with all outputs
    """
    print("\n" + "="*60)
    print("EVALUATION PIPELINE")
    print("="*60)
    
    # Ensure outputs directory exists
    output_dir = BASE_DIR / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'metrics': metrics,
        'predictions': predictions
    }
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics
    save_metrics_to_csv(metrics)
    
    # Get actual and predicted values
    if 'Actual' in predictions.columns and 'Predicted' in predictions.columns:
        y_true = predictions['Actual'].values
        y_pred = predictions['Predicted'].values
    else:
        print("Warning: No actual/predicted values found in predictions DataFrame.")
        return results
    
    # Plot 1: Actual vs Predicted
    print("\nGenerating predictions plot...")
    plot_predictions(
        y_true, y_pred,
        title="PJM East Energy Consumption: Actual vs Predicted",
        save_path=output_dir / 'predictions_plot.png'
    )
    results['predictions_plot'] = output_dir / 'predictions_plot.png'
    
    # Plot 2: Residuals
    print("Generating residuals plot...")
    plot_residuals(
        y_true, y_pred,
        title="Residuals Analysis - Energy Consumption Forecast",
        save_path=output_dir / 'residuals_plot.png'
    )
    results['residuals_plot'] = output_dir / 'residuals_plot.png'
    
    # Plot 3: Scatter plot
    print("Generating scatter plot...")
    plot_scatter_actual_vs_predicted(
        y_true, y_pred,
        title="Actual vs Predicted Energy Consumption",
        save_path=output_dir / 'scatter_plot.png'
    )
    results['scatter_plot'] = output_dir / 'scatter_plot.png'
    
    # Plot 4: Feature Importance - FIXED
    print("Generating feature importance plot...")
    
    # Get feature importance from model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Get feature columns
        if feature_cols is None:
            # Try to get from predictions or use generic names
            if 'feature_cols' in locals():
                pass
            else:
                # Get feature columns from the model's feature names if available
                try:
                    if hasattr(model, 'get_booster'):
                        feature_names = model.get_booster().feature_names
                        if feature_names and len(feature_names) == len(importances):
                            feature_cols = feature_names
                        else:
                            # Create generic feature names
                            feature_cols = [f'Feature_{i}' for i in range(len(importances))]
                    else:
                        feature_cols = [f'Feature_{i}' for i in range(len(importances))]
                except:
                    feature_cols = [f'Feature_{i}' for i in range(len(importances))]
        
        # Ensure lengths match
        if len(feature_cols) != len(importances):
            print(f"Warning: Feature count ({len(feature_cols)}) doesn't match importance count ({len(importances)}). Using generic names.")
            feature_cols = [f'Feature_{i}' for i in range(len(importances))]
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Save to CSV
        importance_path = output_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance saved to: {importance_path}")
        
        # Plot
        plot_feature_importance(
            importance_df,
            top_n=10,
            title="Energy Consumption Forecasting - Feature Importance",
            save_path=output_dir / 'feature_importance.png'
        )
        results['feature_importance_plot'] = output_dir / 'feature_importance.png'
        results['feature_importance_df'] = importance_df
    else:
        print("Warning: Model does not have feature_importances_ attribute.")
    
    # Save comparison metrics
    comparison_df = pd.DataFrame({
        'Model': ['XGBoost'],
        'MAPE': [metrics.get('MAPE', 0)],
        'RMSE': [metrics.get('RMSE', 0)],
        'MAE': [metrics.get('MAE', 0)],
        'R2': [metrics.get('R2', 0)]
    })
    comparison_path = output_dir / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Model comparison saved to: {comparison_path}")
    
    print("\n✅ Evaluation complete!")
    print(f"All outputs saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    # Quick test when run directly
    print("Testing evaluation module...")
    
    # Generate synthetic data for testing
    np.random.seed(42)
    y_true = np.random.randn(100) * 100 + 500
    y_pred = y_true + np.random.randn(100) * 20
    
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics)
    
    print("\n✅ Evaluation module working correctly!")
