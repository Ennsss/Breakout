"""Model evaluation: metrics, ensemble, calibration, cross-fold summary."""

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """Precision in the top-K predicted players.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        k: Number of top predictions to consider

    Returns:
        Precision among top-K predictions
    """
    if k <= 0 or len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    return float(np.sum(y_true[top_k_idx]) / k)


def recall_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """Recall in the top-K predicted players.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        k: Number of top predictions to consider

    Returns:
        Recall among top-K predictions
    """
    total_positives = np.sum(y_true)
    if total_positives == 0 or k <= 0 or len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    return float(np.sum(y_true[top_k_idx]) / total_positives)


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, k_values: list[int] | None = None
) -> dict:
    """Compute all evaluation metrics.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        k_values: List of K values for Precision@K/Recall@K

    Returns:
        Dictionary with all metric values
    """
    if k_values is None:
        k_values = [10, 20, 50, 100]

    metrics = {}

    # ROC-AUC (needs at least one positive and one negative)
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")

    # Average precision
    if len(np.unique(y_true)) >= 2:
        metrics["average_precision"] = float(average_precision_score(y_true, y_proba))
    else:
        metrics["average_precision"] = float("nan")

    # Brier score
    metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))

    # Precision@K and Recall@K
    for k in k_values:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, y_proba, k)
        metrics[f"recall_at_{k}"] = recall_at_k(y_true, y_proba, k)

    return metrics


def ensemble_predictions(
    lgbm_proba: np.ndarray,
    xgb_proba: np.ndarray,
    weights: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """Combine predictions via weighted average.

    Args:
        lgbm_proba: LightGBM predicted probabilities
        xgb_proba: XGBoost predicted probabilities
        weights: (lgbm_weight, xgb_weight), should sum to 1.0

    Returns:
        Weighted average probabilities
    """
    w_lgbm, w_xgb = weights
    return w_lgbm * lgbm_proba + w_xgb * xgb_proba


def calibrate_probabilities(
    y_val: np.ndarray,
    proba_val: np.ndarray,
    proba_test: np.ndarray,
    method: str = "isotonic",
) -> tuple[IsotonicRegression, np.ndarray]:
    """Calibrate probabilities using isotonic regression.

    Fits calibrator on validation set, applies to test set.

    Args:
        y_val: Validation true labels
        proba_val: Validation predicted probabilities
        proba_test: Test predicted probabilities to calibrate
        method: Calibration method ("isotonic" or "platt")

    Returns:
        Tuple of (fitted calibrator, calibrated test probabilities)
    """
    if method == "isotonic":
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(proba_val, y_val)
        calibrated = calibrator.predict(proba_test)
    else:
        # Platt scaling via logistic regression
        from sklearn.linear_model import LogisticRegression

        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(proba_val.reshape(-1, 1), y_val)
        calibrated = calibrator.predict_proba(proba_test.reshape(-1, 1))[:, 1]

    # Clip to [0, 1]
    calibrated = np.clip(calibrated, 0.0, 1.0)

    logger.info(f"Calibrated {len(proba_test)} predictions (method={method})")
    return calibrator, calibrated


def evaluate_fold(
    lgbm_model,
    xgb_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    meta_test: pd.DataFrame,
    config: dict | None = None,
) -> dict:
    """Full evaluation for one fold: ensemble -> calibrate -> metrics.

    Args:
        lgbm_model: Trained LightGBM Booster
        xgb_model: Trained XGBoost Booster
        X_test: Test features
        y_test: Test labels
        X_val: Validation features (for calibration fitting)
        y_val: Validation labels
        meta_test: Metadata DataFrame for test set
        config: Config dict for ensemble weights, calibration, k_values

    Returns:
        Dictionary with metrics, predictions DataFrame, and raw probabilities
    """
    import xgboost as xgb_lib

    config = config or {}
    ensemble_cfg = config.get("ensemble", {})
    eval_cfg = config.get("evaluation", {})

    lgbm_weight = ensemble_cfg.get("weights", {}).get("lightgbm", 0.5)
    xgb_weight = ensemble_cfg.get("weights", {}).get("xgboost", 0.5)
    cal_method = ensemble_cfg.get("calibration", {}).get("method", "isotonic")
    k_values = eval_cfg.get("k_values", [10, 20, 50, 100])

    # Get individual model predictions
    lgbm_proba_test = lgbm_model.predict(X_test)
    dtest = xgb_lib.DMatrix(X_test)
    xgb_proba_test = xgb_model.predict(dtest)

    lgbm_proba_val = lgbm_model.predict(X_val)
    dval = xgb_lib.DMatrix(X_val)
    xgb_proba_val = xgb_model.predict(dval)

    # Ensemble
    ensemble_test = ensemble_predictions(lgbm_proba_test, xgb_proba_test, (lgbm_weight, xgb_weight))
    ensemble_val = ensemble_predictions(lgbm_proba_val, xgb_proba_val, (lgbm_weight, xgb_weight))

    # Calibrate
    calibrator, calibrated_test = calibrate_probabilities(
        y_val, ensemble_val, ensemble_test, method=cal_method
    )

    # Compute metrics for each model
    result = {
        "lgbm_metrics": compute_metrics(y_test, lgbm_proba_test, k_values),
        "xgb_metrics": compute_metrics(y_test, xgb_proba_test, k_values),
        "ensemble_metrics": compute_metrics(y_test, ensemble_test, k_values),
        "calibrated_metrics": compute_metrics(y_test, calibrated_test, k_values),
    }

    # Build predictions DataFrame
    predictions = meta_test.copy()
    predictions["label"] = y_test
    predictions["prob_lgbm"] = lgbm_proba_test
    predictions["prob_xgb"] = xgb_proba_test
    predictions["prob_ensemble"] = ensemble_test
    predictions["prob_calibrated"] = calibrated_test

    result["predictions"] = predictions
    result["calibrator"] = calibrator

    logger.info(f"Fold evaluation: ROC-AUC={result['calibrated_metrics']['roc_auc']:.4f}")
    return result


def cross_fold_summary(fold_results: list[dict]) -> dict:
    """Compute mean and std of metrics across folds.

    Args:
        fold_results: List of evaluate_fold results (one per fold)

    Returns:
        Dictionary with mean/std for each metric
    """
    summary = {}

    for model_key in ["lgbm_metrics", "xgb_metrics", "ensemble_metrics", "calibrated_metrics"]:
        metrics_across_folds = [r[model_key] for r in fold_results]
        metric_names = metrics_across_folds[0].keys()

        model_summary = {}
        for metric in metric_names:
            values = [m[metric] for m in metrics_across_folds if not np.isnan(m[metric])]
            if values:
                model_summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
            else:
                model_summary[metric] = {"mean": float("nan"), "std": float("nan"), "values": []}

        summary[model_key] = model_summary

    return summary
