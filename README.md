# ClusterImputer

[![PyPI version](https://img.shields.io/pypi/v/cluster-imputer.svg)](https://pypi.org/project/cluster-imputer/)
[![Python versions](https://img.shields.io/pypi/pyversions/cluster-imputer.svg)](https://pypi.org/project/cluster-imputer/)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41598--026--70170--9-blue.svg)](https://doi.org/10.1038/s41598-026-70170-9)
[![Journal: Scientific Reports](https://img.shields.io/badge/Nature-Scientific%20Reports-006699.svg)](https://doi.org/10.1038/s41598-026-70170-9)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn Compatible](https://img.shields.io/badge/scikit--learn-compatible-orange.svg)](https://scikit-learn.org/)

A high-performance, scikit-learn compatible **Cluster-Aware Tabular Imputer** for mixed numerical and categorical datasets, published in ***Nature Scientific Reports***.

📖 **Paper:** [Cluster-aware imputation: a hybrid XGBoost-based model for missing data](https://doi.org/10.1038/s41598-026-70170-9)  
🏛️ **Journal:** *Scientific Reports* (Springer Nature), Vol. 16, Art. 70170 (2026)  
🔗 **DOI:** [10.1038/s41598-026-70170-9](https://doi.org/10.1038/s41598-026-70170-9)

---

## 🌟 Overview

`ClusterImputer` (also available as `ClusteringImputer`) addresses the limitation of traditional MICE global regression models by learning **local correlation structures** within tabular data:

1. **Missingness-Aware Chaining**: Features are prioritized and imputed in ascending order of missingness (least missing to most missing).
2. **Subspace Clustering**: For each feature, complete rows are projected into PCA space and partitioned using K-Means with automatic silhouette score optimization ($k$).
3. **Parallel Localized Ensembles**: High-speed XGBoost regressors and classifiers are fitted per cluster in parallel.
4. **Iterative MICE Convergence**: Chained updates repeat iteratively until numerical tolerance or maximum iterations are satisfied.
5. **Linear Computational Scaling**: Benchmarked across **97 UCI datasets**, demonstrating asymptotic linearity $\mathcal{O}(N)$ in computational complexity compared to KNN's quadratic $\mathcal{O}(N^2)$ scaling.

---

## 🚀 Installation

```bash
pip install cluster-imputer
```

### Dependencies
- `numpy >= 1.20.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.0.0`
- `xgboost >= 1.5.0`
- `joblib >= 1.1.0`

---

## 💡 Quick Start

```python
import numpy as np
import pandas as pd
from cluster_imputer import ClusterImputer

# 1. Prepare sample data with missing entries
df = pd.DataFrame({
    "feature_1": [1.2, np.nan, 3.4, 4.1, 5.0, 2.8, np.nan, 3.9],
    "feature_2": [10.5, 20.1, np.nan, 40.2, 51.0, 22.3, 31.1, 41.5],
    "category":  ["A", "B", "A", np.nan, "B", "A", "B", np.nan],
})

# 2. Initialize the imputer
imputer = ClusterImputer(
    n_clusters='auto',   # Automatically finds optimal k via silhouette score
    max_k=5,
    max_iter=5,
    n_jobs=-1
)

# 3. Fit and transform
df_imputed = imputer.fit_transform(df)
print(df_imputed)
```

---

## 🔗 Integration with Scikit-Learn Pipelines

`ClusterImputer` inherits from `BaseEstimator` and `TransformerMixin`, making it seamlessly compatible with `Pipeline`, `ColumnTransformer`, and cross-validation:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from cluster_imputer import ClusterImputer

pipe = Pipeline([
    ("imputer", ClusterImputer(n_clusters=3, max_iter=5, n_jobs=-1)),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

---

## ⚙️ Hyperparameter Auto-Tuning

Not sure which parameters to choose for your dataset? Use `suggest_imputer_params`:

```python
from cluster_imputer import suggest_imputer_params

params = suggest_imputer_params(df)
print("Recommended parameters:", params)

imputer = ClusterImputer(**params)
df_clean = imputer.fit_transform(df)
```

---

## 📖 Parameters & API Reference

### `ClusterImputer` / `ClusteringImputer`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_clusters` | `'auto'` or `int` | `'auto'` | Number of clusters. When `'auto'`, uses silhouette analysis. |
| `max_k` | `int` | `10` | Maximum number of clusters evaluated when `n_clusters='auto'`. |
| `n_pca_components` | `int` | `4` | Number of PCA dimensions for clustering. |
| `max_iter` | `int` | `10` | Maximum outer chained MICE iterations. |
| `tol` | `float` | `1e-3` | Convergence tolerance on squared change of numeric imputed values. |
| `n_jobs` | `int` | `-1` | Number of CPU cores to use for parallel XGBoost training and silhouette computation. |
| `max_training_samples`| `int` or `None`| `10000` | Subsampling limit to guarantee speed and memory safety on large datasets. |

### Methods

- **`fit(X, y=None)`**: Learns clustering spaces and per-cluster XGBoost models on observed data.
- **`transform(X)`**: Imputes missing values in `X` and returns a completed DataFrame or numpy ndarray.
- **`fit_transform(X, y=None)`**: Fits and transforms the data in a single call.

---

## 📄 Citation

If you use `ClusterImputer` in your research, please cite our article in ***Scientific Reports***:

```bibtex
@article{ismail2026clusteraware,
  title = {Cluster-aware imputation: a hybrid XGBoost-based model for missing data},
  author = {Ismail, Mahmoud M. and Emad, Mohamed and Mohamed, Mai},
  journal = {Scientific Reports},
  volume = {16},
  number = {1},
  pages = {70170},
  year = {2026},
  publisher = {Springer Nature},
  doi = {10.1038/s41598-026-70170-9},
  url = {https://doi.org/10.1038/s41598-026-70170-9}
}
```

---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
