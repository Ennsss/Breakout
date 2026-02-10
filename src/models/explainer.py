"""SHAP explanations for model predictions."""

import logging

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_shap_values(
    model, X: np.ndarray, feature_names: list[str], max_samples: int = 1000
) -> shap.Explanation:
    """Compute SHAP values using TreeExplainer.

    Args:
        model: Trained LightGBM or XGBoost model
        X: Feature matrix
        feature_names: List of feature names
        max_samples: Max samples to explain (for performance)

    Returns:
        shap.Explanation object
    """
    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
        indices = np.arange(len(X))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Ensure feature names are set
    shap_values.feature_names = feature_names

    logger.info(f"SHAP values computed for {len(X_sample)} samples, {len(feature_names)} features")
    return shap_values


def feature_importance(shap_values: shap.Explanation, feature_names: list[str]) -> pd.DataFrame:
    """Compute mean absolute SHAP value per feature.

    Args:
        shap_values: SHAP Explanation object
        feature_names: List of feature names

    Returns:
        DataFrame with columns [feature, importance] sorted descending
    """
    # Handle both single-output and multi-output SHAP values
    values = shap_values.values
    if values.ndim == 3:
        # Multi-output: take the positive class (index 1)
        values = values[:, :, 1]

    mean_abs = np.mean(np.abs(values), axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def explain_player(
    shap_values: shap.Explanation,
    player_idx: int,
    feature_names: list[str],
    meta: pd.DataFrame | None = None,
    top_n: int = 10,
) -> dict:
    """Get top features driving a specific player's prediction.

    Args:
        shap_values: SHAP Explanation object
        player_idx: Index of the player in the SHAP values array
        feature_names: List of feature names
        meta: Optional metadata DataFrame
        top_n: Number of top features to return

    Returns:
        Dictionary with player info and top contributing features
    """
    values = shap_values.values
    if values.ndim == 3:
        player_shap = values[player_idx, :, 1]
    else:
        player_shap = values[player_idx]

    # Sort by absolute SHAP value
    abs_shap = np.abs(player_shap)
    top_indices = np.argsort(abs_shap)[::-1][:top_n]

    top_features = []
    for idx in top_indices:
        top_features.append({
            "feature": feature_names[idx],
            "shap_value": float(player_shap[idx]),
            "feature_value": float(shap_values.data[player_idx, idx]),
        })

    result = {
        "player_idx": player_idx,
        "base_value": float(shap_values.base_values[player_idx])
                      if np.ndim(shap_values.base_values) > 0
                      else float(shap_values.base_values),
        "top_features": top_features,
    }

    if meta is not None and player_idx < len(meta):
        for col in ["player_id", "name", "team", "league", "season"]:
            if col in meta.columns:
                result[col] = str(meta.iloc[player_idx][col])

    return result


def generate_explanations(
    lgbm_model,
    xgb_model,
    X_test: np.ndarray,
    feature_names: list[str],
    meta_test: pd.DataFrame | None = None,
    max_samples: int = 1000,
) -> dict:
    """Full SHAP pipeline for one fold.

    Computes SHAP for both models and produces aggregated feature importance.

    Args:
        lgbm_model: Trained LightGBM model
        xgb_model: Trained XGBoost model
        X_test: Test features
        feature_names: Feature names
        meta_test: Test set metadata
        max_samples: Max samples for SHAP computation

    Returns:
        Dictionary with SHAP values, feature importance, and per-model explanations
    """
    lgbm_shap = compute_shap_values(lgbm_model, X_test, feature_names, max_samples)
    xgb_shap = compute_shap_values(xgb_model, X_test, feature_names, max_samples)

    lgbm_importance = feature_importance(lgbm_shap, feature_names)
    xgb_importance = feature_importance(xgb_shap, feature_names)

    # Average importance across models
    merged = lgbm_importance.merge(
        xgb_importance, on="feature", suffixes=("_lgbm", "_xgb")
    )
    merged["importance"] = (merged["importance_lgbm"] + merged["importance_xgb"]) / 2
    merged = merged.sort_values("importance", ascending=False).reset_index(drop=True)

    result = {
        "lgbm_shap": lgbm_shap,
        "xgb_shap": xgb_shap,
        "lgbm_importance": lgbm_importance,
        "xgb_importance": xgb_importance,
        "combined_importance": merged[["feature", "importance"]],
    }

    logger.info(f"Explanations generated: top feature = {merged.iloc[0]['feature']}")
    return result
