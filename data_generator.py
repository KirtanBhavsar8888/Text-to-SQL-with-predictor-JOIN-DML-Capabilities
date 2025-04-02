import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 10k samples
n_samples = 10000

# Feature columns
data = {
    # Numerical features
    'sales_volume': np.random.normal(50, 15, n_samples).round(2),
    'temperature': np.random.uniform(-10, 35, n_samples).round(1),
    
    # Categorical features
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    'product_type': np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Furniture'], n_samples),
    
    # Time series feature
    'date': pd.date_range(start='2023-01-01', periods=n_samples, freq='H'),
    
    # Target columns
    'regression_target': np.random.normal(0, 5, n_samples).cumsum() + 100,  # Simulate trend
    'classification_target': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.2, 0.5, 0.3]),
    'time_series_target': (np.sin(np.linspace(0, 20*np.pi, n_samples)) * 50 + 100) + np.random.normal(0, 5, n_samples)
}

df = pd.DataFrame(data)

# Add noise to regression target based on features
df['regression_target'] += df['sales_volume'] * 0.8 + df['temperature'] * 0.5

# Save to CSV
df.to_csv("synthetic_dataset.csv", index=False)