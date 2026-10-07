"""
ClusterImputer: A high-performance cluster-then-predict missing data imputer
for mixed numerical and categorical tabular data.
"""

from .imputer import (
    ClusterImputer,
    ClusteringImputer,
    suggest_imputer_params,
    _train_single_model,
)
from .utils import (
    compute_silhouette_scores_parallel,
    compute_silhouette_scores,
    find_optimal_k,
    compute_wcss,
)

__version__ = "0.1.1"
__author__ = "Mohamed Emad"
__email__ = "mohamed.emad222210@gmail.com"

__all__ = [
    "ClusterImputer",
    "ClusteringImputer",
    "suggest_imputer_params",
    "_train_single_model",
    "compute_silhouette_scores_parallel",
    "compute_silhouette_scores",
    "find_optimal_k",
    "compute_wcss",
    "__version__",
]
