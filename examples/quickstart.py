"""
Quickstart example demonstrating ClusterImputer on a mixed tabular dataset.
"""

import numpy as np
import pandas as pd
from cluster_imputer import ClusterImputer, suggest_imputer_params

# 1. Create a synthetic mixed tabular dataset
np.random.seed(42)
n_samples = 200

data = {
    "age": np.random.normal(loc=35, scale=10, size=n_samples),
    "income": np.random.normal(loc=50000, scale=15000, size=n_samples),
    "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], size=n_samples),
    "credit_score": np.random.normal(loc=700, scale=50, size=n_samples),
    "city": np.random.choice(["Cairo", "Alexandria", "Giza"], size=n_samples),
}

df = pd.DataFrame(data)

# 2. Inject missing values (NaN)
df.loc[np.random.choice(n_samples, size=30, replace=False), "age"] = np.nan
df.loc[np.random.choice(n_samples, size=25, replace=False), "income"] = np.nan
df.loc[np.random.choice(n_samples, size=20, replace=False), "education"] = np.nan

print("Original missing values count per column:")
print(df.isnull().sum())

# 3. (Optional) Auto-suggest hyperparameters based on dataset properties
suggested = suggest_imputer_params(df)
print("\nSuggested Hyperparameters:", suggested)

# 4. Initialize and fit ClusterImputer
imputer = ClusterImputer(
    n_clusters='auto',       # Auto-determine optimal k via silhouette analysis
    max_k=6,
    n_pca_components=3,
    max_iter=5,
    n_jobs=-1,
)

print("\nFitting and imputing...")
df_imputed = imputer.fit_transform(df)

print("\nMissing values after imputation:")
print(df_imputed.isnull().sum())

print("\nFirst 5 rows of imputed dataset:")
print(df_imputed.head())
