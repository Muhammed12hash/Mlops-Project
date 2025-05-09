import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import time
from IPython.display import clear_output

def load_monitoring_data():
    """Load monitoring data from the metrics log file."""
    monitoring_file = os.path.join("model_monitoring", "metrics_log.jsonl")
    
    if not os.path.exists(monitoring_file):
        return pd.DataFrame()
    
    # Read the JSONL file
    data = []
    with open(monitoring_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def plot_metrics(df, window_size=10):
    """Plot monitoring metrics over time."""
    if df.empty:
        print("No monitoring data available yet.")
        return
    
    # Calculate rolling averages
    rolling_metrics = df.set_index('timestamp').rolling(window=window_size).mean()
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot each metric
    metrics = ['roc_auc', 'precision', 'recall', 'f1']
    for metric in metrics:
        plt.plot(rolling_metrics.index, rolling_metrics[metric], 
                label=f'{metric} (rolling avg)', marker='o', markersize=4)
    
    plt.title(f'Model Performance Metrics Over Time\n(Rolling Average Window: {window_size})')
    plt.xlabel('Timestamp')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Add threshold lines
    plt.axhline(y=0.80, color='r', linestyle='--', alpha=0.3, label='Minimum Threshold')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join("model_monitoring", "performance_metrics.png"))
    plt.close()

def calculate_drift_metrics(df):
    """Calculate drift metrics from monitoring data."""
    if df.empty:
        return None
    
    # Calculate basic drift metrics
    recent_window = df.iloc[-10:] if len(df) > 10 else df
    baseline_window = df.iloc[:10] if len(df) > 10 else df
    
    drift_metrics = {}
    for metric in ['roc_auc', 'precision', 'recall', 'f1']:
        baseline_mean = baseline_window[metric].mean()
        recent_mean = recent_window[metric].mean()
        drift = recent_mean - baseline_mean
        drift_metrics[metric] = {
            'baseline_mean': baseline_mean,
            'recent_mean': recent_mean,
            'drift': drift,
            'drift_percentage': (drift / baseline_mean) * 100 if baseline_mean != 0 else 0
        }
    
    return drift_metrics

def print_summary(df, drift_metrics):
    """Print monitoring summary."""
    if df.empty:
        print("No monitoring data available yet.")
        return
    
    print("\n=== Model Monitoring Summary ===")
    print(f"Last Update: {df['timestamp'].max()}")
    print(f"Total Predictions: {len(df)}")
    
    print("\nCurrent Metrics (Last Batch):")
    last_metrics = df.iloc[-1]
    for metric in ['roc_auc', 'precision', 'recall', 'f1']:
        print(f"{metric}: {last_metrics[metric]:.4f}")
    
    if drift_metrics:
        print("\nDrift Analysis:")
        for metric, values in drift_metrics.items():
            print(f"\n{metric}:")
            print(f"  Baseline: {values['baseline_mean']:.4f}")
            print(f"  Recent:   {values['recent_mean']:.4f}")
            print(f"  Drift:    {values['drift']:.4f} ({values['drift_percentage']:.2f}%)")

def main():
    print("Starting monitoring dashboard...")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            # Load monitoring data
            df = load_monitoring_data()
            
            if not df.empty:
                # Calculate drift metrics
                drift_metrics = calculate_drift_metrics(df)
                
                # Update visualizations
                plot_metrics(df)
                
                # Clear terminal and print summary
                clear_output(wait=True)
                print_summary(df, drift_metrics)
            
            # Update every 60 seconds
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nStopping monitoring dashboard...")

if __name__ == "__main__":
    main() 