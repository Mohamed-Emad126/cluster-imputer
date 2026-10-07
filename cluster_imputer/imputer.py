"""
ClusterImputer: Chained Cluster-then-Predict Tabular Missing Data Imputer.

A novel cluster-then-predict methodology:
1. Imputes features in order of least-to-most missingness.
2. For each feature, clusters complete rows using PCA + K-Means (optimal k via silhouette analysis).
3. Fits a specialized XGBoost regressor/classifier for each cluster in parallel.
4. Predicts and imputes missing entries using the cluster-assigned models.
5. Iterates until numeric convergence (MICE-style chaining).
"""

import warnings
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .utils import compute_silhouette_scores_parallel, find_optimal_k


def _train_single_model(
    cluster_id: int,
    X_train_cluster: np.ndarray,
    y_train_cluster: pd.Series,
    is_categorical: bool,
    global_fallback_value: Any
) -> Tuple[int, Any]:
    """
    Trains a single XGBoost prediction model for a specific cluster's data.
    Designed to be called in parallel by joblib.
    """
    try:
        if isinstance(y_train_cluster, pd.Series):
            notna_mask = y_train_cluster.notna().values
            if not isinstance(X_train_cluster, np.ndarray):
                X_train_cluster = np.asarray(X_train_cluster)
            if notna_mask.shape[0] == X_train_cluster.shape[0]:
                X_train_cluster = X_train_cluster[notna_mask]
                y_train_cluster = y_train_cluster[notna_mask]
            else:
                return cluster_id, global_fallback_value
        else:
            y_train_cluster = pd.Series(y_train_cluster)
            notna_mask = y_train_cluster.notna().values
            if notna_mask.shape[0] == X_train_cluster.shape[0]:
                X_train_cluster = X_train_cluster[notna_mask]
                y_train_cluster = y_train_cluster[notna_mask]
            else:
                return cluster_id, global_fallback_value

        if X_train_cluster is None or len(X_train_cluster) < 3:
            return cluster_id, global_fallback_value

        common_params = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method='hist',
            n_jobs=-1,
            random_state=42,
        )

        if is_categorical:
            if y_train_cluster.nunique(dropna=True) < 2:
                local_mode = y_train_cluster.mode(dropna=True)[0] if not y_train_cluster.empty else global_fallback_value
                return cluster_id, local_mode

            y_cat = y_train_cluster.astype('category')
            y_train_encoded = pd.Categorical(y_cat).codes
            categories = list(pd.Categorical(y_cat).categories)
            categories_map = dict(enumerate(categories))
            num_class = int(len(categories))

            if num_class == 2:
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    **common_params
                )
            else:
                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    eval_metric='mlogloss',
                    num_class=num_class,
                    **common_params
                )
            model.fit(X_train_cluster, y_train_encoded)
            return cluster_id, (model, categories_map)
        else:
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                **common_params
            )
            model.fit(X_train_cluster, y_train_cluster)
            return cluster_id, model
    except Exception as e:
        warnings.warn(f"Error training model for cluster {cluster_id}: {e}")
        return cluster_id, global_fallback_value


class ClusteringImputer(BaseEstimator, TransformerMixin):
    """
    Cluster-then-predict imputer for mixed numerical and categorical tabular data.

    Parameters
    ----------
    n_clusters : 'auto' or int, default='auto'
        Number of clusters to form. If 'auto', uses silhouette score optimization.
    max_k : int, default=10
        Maximum clusters to evaluate when n_clusters='auto'.
    n_pca_components : int, default=4
        Number of PCA dimensions for clustering.
    max_iter : int, default=10
        Maximum outer MICE-style chained iterations.
    tol : float, default=1e-3
        Convergence tolerance on squared change of numeric imputed values.
    n_jobs : int, default=-1
        Parallel threads for XGBoost and silhouette search.
    max_training_samples : int, default=10000
        Maximum sample rows used to fit clustering to protect memory and speed.
    """

    def __init__(
        self,
        n_clusters: Any = 'auto',
        max_k: int = 10,
        n_pca_components: int = 4,
        max_iter: int = 10,
        tol: float = 1e-3,
        n_jobs: int = -1,
        max_training_samples: Optional[int] = 10000,
    ):
        self.n_clusters = n_clusters
        self.max_k = max_k
        self.n_pca_components = n_pca_components
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.max_training_samples = max_training_samples

        self.imputation_order_: List[str] = []
        self.trained_models_: Dict[str, Any] = {}
        self._categorical_features: List[str] = []
        self._feature_columns: List[str] = []
        self._simple_imputer = SimpleImputer(strategy='mean')

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Learns clustering models and prediction estimators per cluster.

        Parameters
        ----------
        X : {array-like, pd.DataFrame} of shape (n_samples, n_features)
            The input data with missing values.
        y : Ignored
            Not used, present for scikit-learn API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        else:
            df = X.copy()

        self.feature_names_in_ = df.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        self._determine_imputation_order(df)
        self._identify_feature_types(df)
        df_imputed_temp = self._preprocess_features(df)

        for target_col in self.imputation_order_:
            is_categorical = target_col in self._categorical_features
            df_missing = df_imputed_temp[df[target_col].isnull()]
            df_complete = df_imputed_temp[~df[target_col].isnull()]
            if df_missing.empty:
                continue

            features = self._get_features_for_target(df_imputed_temp, target_col, is_categorical)
            if not features:
                imputer_strat = 'most_frequent' if is_categorical else 'mean'
                simple_imp = SimpleImputer(strategy=imputer_strat)
                simple_imp.fit(df.loc[~df[target_col].isnull(), [target_col]])
                self.trained_models_[target_col] = {'simple_imputer': simple_imp, 'features': []}
                imputed_values = simple_imp.transform(df.loc[df[target_col].isnull(), [target_col]]).ravel()
                df.loc[df[target_col].isnull(), target_col] = imputed_values
                df_imputed_temp = self._preprocess_features(df)
                continue

            scaler, pca, kmeans_model = self._train_clustering_model(df_complete, features)
            prediction_models = self._train_prediction_models(
                df, df_complete, features, target_col, is_categorical, scaler, pca, kmeans_model
            )
            self.trained_models_[target_col] = {
                'scaler': scaler,
                'pca': pca,
                'kmeans': kmeans_model,
                'prediction_models': prediction_models,
                'features': features
            }
            imputed_values = self._impute_values(
                df, df_missing, self.trained_models_[target_col], target_col, is_categorical
            )
            df.loc[df[target_col].isnull(), target_col] = imputed_values.to_numpy()
            df_imputed_temp = self._preprocess_features(df)

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Imputes all missing values in X using fitted cluster models.

        Parameters
        ----------
        X : {array-like, pd.DataFrame} of shape (n_samples, n_features)
            The input data with missing values to be imputed.

        Returns
        -------
        X_imputed : {pd.DataFrame, np.ndarray}
            Data with imputed values, retaining the input type.
        """
        check_is_fitted(self, 'trained_models_')
        is_numpy = not isinstance(X, pd.DataFrame)
        if is_numpy:
            df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            df = X.copy()

        missing_mask = df.isnull()

        # Initial simple imputation fallback
        for col in self.imputation_order_:
            if not missing_mask[col].any():
                continue
            is_cat = col in self._categorical_features
            strategy = 'most_frequent' if is_cat else 'mean'
            imp = SimpleImputer(strategy=strategy)
            imputed_initial = imp.fit_transform(df.loc[~missing_mask[col], [col]]).ravel()
            df.loc[missing_mask[col], col] = imputed_initial[0] if len(imputed_initial) > 0 else np.nan

        # Chained iterative imputation
        for _ in range(self.max_iter):
            df_last_iter = df.copy()
            for target_col in self.imputation_order_:
                if not missing_mask[target_col].any():
                    continue
                models = self.trained_models_.get(target_col)
                if not models:
                    continue

                df_processed = self._preprocess_features(df)
                df_to_impute = df_processed[missing_mask[target_col]]

                if 'simple_imputer' in models:
                    imputed_vals = models['simple_imputer'].transform(df.loc[missing_mask[target_col], [target_col]]).ravel()
                    df.loc[missing_mask[target_col], target_col] = imputed_vals
                else:
                    imputed_vals = self._impute_values(
                        df, df_to_impute, models, target_col, is_categorical=target_col in self._categorical_features
                    )
                    df.loc[missing_mask[target_col], target_col] = imputed_vals.to_numpy()

            numeric_cols_imputed = [col for col in self.imputation_order_ if col not in self._categorical_features]
            if numeric_cols_imputed:
                change = float(np.sum((df[numeric_cols_imputed].values - df_last_iter[numeric_cols_imputed].values) ** 2))
                if change < self.tol:
                    break

        return df.to_numpy() if is_numpy else df

    def _determine_imputation_order(self, df: pd.DataFrame):
        missing_counts = df.isnull().sum()
        self.imputation_order_ = missing_counts[missing_counts > 0].sort_values().index.tolist()

    def _identify_feature_types(self, df: pd.DataFrame):
        self._categorical_features = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(df, columns=self._categorical_features, dummy_na=False, drop_first=True)

    def _get_features_for_target(self, df_processed: pd.DataFrame, target_col: str, is_categorical: bool) -> List[str]:
        features = [col for col in df_processed.select_dtypes(include=np.number).columns if col != target_col]
        if is_categorical:
            features = [f for f in features if not f.startswith(f"{target_col}_")]
        return features

    def _train_clustering_model(self, df_complete: pd.DataFrame, features: List[str]) -> Tuple[StandardScaler, PCA, KMeans]:
        df_for_fitting = df_complete[features].dropna()
        if self.max_training_samples is not None and len(df_for_fitting) > self.max_training_samples:
            df_for_fitting = df_for_fitting.sample(n=self.max_training_samples, random_state=42)

        n_samples, n_features = df_for_fitting.shape
        if n_features == 0 or n_samples < 2:
            scaler = StandardScaler()
            pca = PCA(n_components=1)
            dummy_data = np.random.randn(max(2, n_samples), max(1, n_features))
            scaler.fit(dummy_data)
            pca.fit(dummy_data)
            kmeans = KMeans(n_clusters=1, random_state=42, n_init='auto').fit(pca.transform(dummy_data))
            return scaler, pca, kmeans

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_for_fitting)
        effective_n_components = min(self.n_pca_components, n_features, n_samples)
        pca = PCA(n_components=effective_n_components)
        X_pca = pca.fit_transform(X_scaled)

        k = self.n_clusters
        if k == 'auto':
            effective_max_k = min(self.max_k, n_samples - 1)
            if effective_max_k < 2:
                k = 1
            else:
                scores = compute_silhouette_scores_parallel(X_pca, effective_max_k, n_jobs=self.n_jobs)
                k = np.argmax(scores) + 2 if scores and max(scores) > -0.5 else 2

        k = min(max(1, int(k)), n_samples)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_pca)
        return scaler, pca, kmeans

    def _train_prediction_models(self, df_original, df_complete, features, target_col, is_categorical, scaler, pca, kmeans) -> Dict:
        df_for_training = df_complete[features].dropna()
        if df_for_training.empty or df_for_training.shape[0] < 2:
            fallback = df_original[target_col].mode()[0] if is_categorical else df_original[target_col].mean()
            return {0: fallback}

        train_idx = df_for_training.index
        y_for_training = df_original.loc[train_idx, target_col]
        X_scaled = scaler.transform(df_for_training)
        X_pca = pca.transform(X_scaled)

        if kmeans.n_clusters == 1:
            cluster_labels = np.zeros(len(df_for_training), dtype=int)
        else:
            cluster_labels = kmeans.predict(X_pca)

        df_training_with_clusters = pd.DataFrame({'cluster': cluster_labels}, index=train_idx)
        df_training_with_clusters['target'] = y_for_training

        if self.max_training_samples and len(df_training_with_clusters) > self.max_training_samples:
            df_training_with_clusters = df_training_with_clusters.sample(n=self.max_training_samples, random_state=42)

        def build_cluster_data(cid: int) -> Tuple[np.ndarray, pd.Series]:
            idx_c = df_training_with_clusters.index[df_training_with_clusters['cluster'] == cid]
            if len(idx_c) == 0:
                return np.zeros((0, pca.n_components)), pd.Series(dtype=float)
            X_c = df_for_training.loc[idx_c]
            X_c_proj = pca.transform(scaler.transform(X_c))
            y_c = y_for_training.loc[idx_c]
            return X_c_proj, y_c

        global_fallback = (
            df_original[target_col].mode(dropna=True)[0] if is_categorical
            else float(df_original[target_col].dropna().mean())
        )

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_train_single_model)(
                cid,
                *build_cluster_data(cid),
                is_categorical,
                global_fallback
            ) for cid in sorted(np.unique(cluster_labels))
        )

        return {cluster_id: model for cluster_id, model in results}

    def _impute_values(self, df_original, df_missing, models, target_col, is_categorical) -> pd.Series:
        if df_missing.empty:
            return pd.Series(dtype=object if is_categorical else float)

        fallback_value = df_original[target_col].mode()[0] if is_categorical else float(df_original[target_col].dropna().mean())
        features = models['features']
        available_features = [f for f in features if f in df_missing.columns]
        if len(available_features) != len(features):
            return pd.Series([fallback_value] * len(df_missing), index=df_missing.index, dtype=object if is_categorical else float)

        df_missing_features = df_missing[features]
        try:
            df_missing_imputed = self._simple_imputer.fit_transform(df_missing_features)
            df_missing_imputed_df = pd.DataFrame(df_missing_imputed, columns=features, index=df_missing.index)
            X_scaled = models['scaler'].transform(df_missing_imputed_df)
            X_pca = models['pca'].transform(X_scaled)
        except Exception:
            return pd.Series([fallback_value] * len(df_missing), index=df_missing.index, dtype=object if is_categorical else float)

        if models['kmeans'].n_clusters == 1 or X_pca.shape[0] == 0:
            missing_clusters = np.zeros(len(df_missing), dtype=int)
        else:
            missing_clusters = models['kmeans'].predict(X_pca)

        if is_categorical:
            imputed_values = pd.Series(index=df_missing.index, dtype=object)
        else:
            imputed_values = pd.Series(index=df_missing.index, dtype=float)

        for cluster_id in np.unique(missing_clusters):
            cluster_mask = (missing_clusters == cluster_id)
            if not np.any(cluster_mask):
                continue
            X_cluster_pca = X_pca[cluster_mask]
            model_or_value = models['prediction_models'].get(cluster_id, fallback_value)

            if is_categorical:
                if isinstance(model_or_value, tuple):
                    model, cat_map = model_or_value
                    pred_probas = model.predict_proba(X_cluster_pca)
                    pred_codes = np.argmax(pred_probas, axis=1)
                    predictions = [cat_map.get(code, fallback_value) for code in pred_codes]
                else:
                    predictions = [model_or_value] * len(X_cluster_pca)
                imputed_values.iloc[np.where(cluster_mask)[0]] = predictions
            else:
                if isinstance(model_or_value, xgb.XGBRegressor):
                    predictions = model_or_value.predict(X_cluster_pca)
                else:
                    predictions = np.full(len(X_cluster_pca), model_or_value, dtype=float)
                imputed_values.iloc[np.where(cluster_mask)[0]] = np.asarray(predictions, dtype=float)

        if not is_categorical:
            imputed_values = pd.to_numeric(imputed_values, errors='coerce').astype(float)

        return imputed_values


# Alias for convenience and backwards compatibility
ClusterImputer = ClusteringImputer


def suggest_imputer_params(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes dataset shape, missingness, and variance to suggest optimal
    hyperparameters (max_k, n_pca_components, max_training_samples, max_iter, tol)
    for ClusterImputer / ClusteringImputer.
    """
    n_samples, n_features = df.shape
    params: Dict[str, Any] = {}

    total_cells = n_samples * max(n_features, 1)
    missing_ratio = float(df.isna().sum().sum()) / total_cells if total_cells > 0 else 0.0

    numeric_cols = df.select_dtypes(include=np.number).columns
    n_num = len(numeric_cols)

    # 1. max_k
    if n_samples < 200:
        base_k = 6
    elif n_samples < 2000:
        base_k = 12
    elif n_samples < 20000:
        base_k = 20
    else:
        base_k = 30

    if missing_ratio > 0.5:
        base_k = max(3, int(base_k * 0.6))
    if n_features > 100:
        base_k = max(5, int(base_k * 0.7))

    base_k = min(base_k, max(2, n_samples - 1))
    params["max_k"] = base_k

    # 2. n_pca_components
    if n_num == 0:
        n_pca = 1
    else:
        raw = max(2, int(n_num / 2))
        if n_samples < 200:
            raw = min(raw, 6)
        elif n_samples < 2000:
            raw = min(raw, 12)
        else:
            raw = min(raw, 20)
        n_pca = min(raw, n_num, max(1, n_samples - 1))
    params["n_pca_components"] = n_pca

    # 3. max_training_samples
    if n_samples <= 5000:
        max_samples = None
    elif n_samples <= 50000:
        max_samples = 10000
    elif n_samples <= 200000:
        max_samples = 20000
    else:
        max_samples = 40000

    if missing_ratio > 0.6 and max_samples is not None:
        max_samples = int(max_samples * 0.7)
    params["max_training_samples"] = max_samples

    # 4. max_iter & tol
    if n_samples < 5000:
        if missing_ratio < 0.2:
            max_iter, tol = 30, 1e-3
        elif missing_ratio < 0.5:
            max_iter, tol = 25, 2e-3
        else:
            max_iter, tol = 20, 3e-3
    elif n_samples < 50000:
        if missing_ratio < 0.2:
            max_iter, tol = 20, 2e-3
        else:
            max_iter, tol = 15, 3e-3
    else:
        max_iter, tol = 10, 5e-3

    params["max_iter"] = max_iter
    params["tol"] = tol
    params["n_clusters"] = "auto"
    params["n_jobs"] = -1

    return params
