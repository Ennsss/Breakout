"""Model training, evaluation, and prediction."""

from src.models.evaluator import (
    calibrate_probabilities,
    compute_metrics,
    cross_fold_summary,
    ensemble_predictions,
    evaluate_fold,
    precision_at_k,
    recall_at_k,
)
from src.models.explainer import (
    compute_shap_values,
    explain_player,
    feature_importance,
    generate_explanations,
)
from src.models.trainer import (
    get_feature_columns,
    load_fold,
    train_baseline,
    train_lgbm,
    train_xgb,
)
from src.models.tuner import tune_lgbm, tune_xgb

__all__ = [
    "calibrate_probabilities",
    "compute_metrics",
    "compute_shap_values",
    "cross_fold_summary",
    "ensemble_predictions",
    "evaluate_fold",
    "explain_player",
    "feature_importance",
    "generate_explanations",
    "get_feature_columns",
    "load_fold",
    "precision_at_k",
    "recall_at_k",
    "train_baseline",
    "train_lgbm",
    "train_xgb",
    "tune_lgbm",
    "tune_xgb",
]
