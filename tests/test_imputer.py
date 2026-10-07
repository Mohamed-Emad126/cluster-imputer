import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from cluster_imputer import ClusterImputer, ClusteringImputer, suggest_imputer_params


def test_alias():
    assert ClusterImputer is ClusteringImputer


def test_suggest_imputer_params():
    df = pd.DataFrame(np.random.randn(100, 5), columns=[f"col_{i}" for i in range(5)])
    df.iloc[0:10, 0] = np.nan
    params = suggest_imputer_params(df)
    assert isinstance(params, dict)
    assert "max_k" in params
    assert "n_pca_components" in params
    assert "max_iter" in params
    assert "tol" in params


def test_numeric_imputation():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(60, 4), columns=["a", "b", "c", "d"])
    X.iloc[5:15, 0] = np.nan
    X.iloc[20:30, 2] = np.nan

    imputer = ClusterImputer(n_clusters=2, max_iter=2, tol=1e-2, n_jobs=1)
    imputer.fit(X)
    X_imp = imputer.transform(X)

    assert isinstance(X_imp, pd.DataFrame)
    assert not X_imp.isnull().any().any()
    assert X_imp.shape == X.shape


def test_mixed_imputation():
    np.random.seed(42)
    X = pd.DataFrame({
        "num1": np.random.randn(50),
        "num2": np.random.randn(50),
        "cat1": np.random.choice(["cat", "dog", "bird"], size=50),
    })
    X.iloc[2:8, 0] = np.nan
    X.iloc[10:15, 2] = np.nan

    imputer = ClusterImputer(n_clusters=2, max_iter=2, tol=1e-2, n_jobs=1)
    X_imp = imputer.fit_transform(X)

    assert isinstance(X_imp, pd.DataFrame)
    assert not X_imp.isnull().any().any()
    assert all(val in ["cat", "dog", "bird"] for val in X_imp["cat1"])


def test_numpy_array_input():
    np.random.seed(42)
    X = np.random.randn(50, 4)
    X[5:10, 0] = np.nan
    X[15:20, 1] = np.nan

    imputer = ClusterImputer(n_clusters=2, max_iter=2, tol=1e-2, n_jobs=1)
    X_imp = imputer.fit_transform(X)

    assert isinstance(X_imp, np.ndarray)
    assert not np.isnan(X_imp).any()
    assert X_imp.shape == X.shape


def test_pipeline_integration():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(60, 4), columns=["x1", "x2", "x3", "x4"])
    y = np.random.choice([0, 1], size=60)
    X.iloc[3:10, 1] = np.nan

    pipe = Pipeline([
        ("imputer", ClusterImputer(n_clusters=2, max_iter=2, tol=1e-2, n_jobs=1)),
        ("classifier", RandomForestClassifier(n_estimators=5, random_state=42)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == 60


def test_no_missing_values():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(30, 3), columns=["f1", "f2", "f3"])
    imputer = ClusterImputer(n_clusters=2, max_iter=2, n_jobs=1)
    X_imp = imputer.fit_transform(X)
    assert not X_imp.isnull().any().any()
    np.testing.assert_allclose(X.values, X_imp.values)
