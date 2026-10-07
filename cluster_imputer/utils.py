"""
Clustering Utilities for Optimal Cluster Selection and Silhouette Analysis.
"""

from typing import List, Optional
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_wcss(X: np.ndarray, max_k: int = 10) -> List[float]:
    """
    Computes within-cluster sum of squares (WCSS / inertia) for k from 1 to max_k.
    """
    wcss = []
    n_samples = X.shape[0]
    for k in range(1, min(max_k, n_samples) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        wcss.append(float(kmeans.inertia_))
    return wcss


def _calculate_score_for_k(k: int, X: np.ndarray) -> float:
    if X is None or X.size == 0 or X.shape[0] <= k:
        return -1.0
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(X)
        n_labels = len(np.unique(cluster_labels))
        if n_labels < 2:
            return -1.0
        return float(silhouette_score(X, cluster_labels))
    except Exception:
        return -1.0


def compute_silhouette_scores_parallel(X: np.ndarray, max_k: int = 10, n_jobs: int = -1) -> List[float]:
    """
    Computes silhouette scores for k in [2, max_k] using parallel execution.
    """
    if X is None or X.size == 0 or X.shape[0] < 3 or X.shape[1] == 0:
        return [-1.0] * max(1, max_k - 1)

    effective_max_k = min(max_k, X.shape[0] - 1)
    if effective_max_k < 2:
        return [-1.0] * max(1, max_k - 1)

    silhouette_scores = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_score_for_k)(k, X) for k in range(2, effective_max_k + 1)
    )
    if len(silhouette_scores) < max_k - 1:
        silhouette_scores.extend([-1.0] * (max_k - 1 - len(silhouette_scores)))
    return silhouette_scores


def compute_silhouette_scores(X: np.ndarray, max_k: int = 10) -> List[float]:
    """Sequential silhouette score computation."""
    return compute_silhouette_scores_parallel(X, max_k=max_k, n_jobs=1)


def find_optimal_k(X: np.ndarray, max_k: int = 10, n_jobs: int = -1) -> int:
    """
    Finds the optimal number of clusters k that maximizes the silhouette score.
    Defaults to 2 if no positive silhouette score is found.
    """
    if X is None or X.shape[0] < 3:
        return 1

    scores = compute_silhouette_scores_parallel(X, max_k=max_k, n_jobs=n_jobs)
    valid_scores = [s for s in scores if s > -1.0]
    if not valid_scores:
        return 1

    optimal_idx = scores.index(max(valid_scores))
    optimal_k = optimal_idx + 2  # k starts at 2
    return optimal_k
