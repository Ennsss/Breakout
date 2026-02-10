"""Tests for Phase 3: model training, evaluation, tuning, and explainability."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers: synthetic data generation
# ---------------------------------------------------------------------------

def _make_synthetic_df(n_samples=200, n_features=10, positive_rate=0.05, seed=42):
    """Create a synthetic DataFrame mimicking fold parquet structure."""
    rng = np.random.RandomState(seed)

    data = {}
    feature_names = [f"feat_{i}" for i in range(n_features)]
    for feat in feature_names:
        data[feat] = rng.randn(n_samples)

    # Metadata columns
    data["player_id"] = [f"p{i}" for i in range(n_samples)]
    data["name"] = [f"Player {i}" for i in range(n_samples)]
    data["team"] = [f"Team {i % 5}" for i in range(n_samples)]
    data["league"] = ["eredivisie"] * n_samples
    data["season"] = ["2020-21"] * n_samples
    data["position"] = ["FW"] * n_samples
    data["position_group"] = ["FW"] * n_samples

    # Label: make it slightly correlated with feat_0 for non-random models
    n_positive = max(1, int(n_samples * positive_rate))
    labels = np.zeros(n_samples)
    # Put positives where feat_0 is highest
    top_indices = np.argsort(data["feat_0"])[-n_positive:]
    labels[top_indices] = 1
    data["label"] = labels

    return pd.DataFrame(data), feature_names


def _make_fold_parquets(tmp_dir, fold_num=1, n_train=200, n_val=50, n_test=50):
    """Write synthetic fold parquet files to tmp_dir."""
    train_df, features = _make_synthetic_df(n_train, seed=42)
    val_df, _ = _make_synthetic_df(n_val, seed=43)
    test_df, _ = _make_synthetic_df(n_test, seed=44)

    train_df.to_parquet(Path(tmp_dir) / f"fold_{fold_num}_train.parquet", index=False)
    val_df.to_parquet(Path(tmp_dir) / f"fold_{fold_num}_val.parquet", index=False)
    test_df.to_parquet(Path(tmp_dir) / f"fold_{fold_num}_test.parquet", index=False)

    return features


def _get_config():
    """Return a minimal config dict matching model_params.yaml structure."""
    return {
        "training": {"random_seed": 42, "n_jobs": 1, "verbose": 0},
        "imbalance": {"strategy": "class_weight", "positive_weight": 10},
        "logistic": {
            "params": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 42,
            }
        },
        "lightgbm": {
            "base_params": {
                "objective": "binary",
                "metric": "auc",
                "num_leaves": 15,
                "learning_rate": 0.1,
                "verbose": -1,
                "random_state": 42,
            },
            "tuning_space": {
                "num_leaves": {"low": 15, "high": 31},
                "learning_rate": {"low": 0.01, "high": 0.2, "log": True},
                "feature_fraction": {"low": 0.5, "high": 1.0},
                "bagging_fraction": {"low": 0.5, "high": 1.0},
                "min_child_samples": {"low": 5, "high": 50},
                "reg_alpha": {"low": 0.0, "high": 1.0},
                "reg_lambda": {"low": 0.0, "high": 1.0},
            },
            "tuning_trials": 5,
            "early_stopping_rounds": 10,
        },
        "xgboost": {
            "base_params": {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 4,
                "learning_rate": 0.1,
                "verbosity": 0,
                "random_state": 42,
            },
            "tuning_space": {
                "max_depth": {"low": 3, "high": 6},
                "learning_rate": {"low": 0.01, "high": 0.2, "log": True},
                "subsample": {"low": 0.5, "high": 1.0},
                "colsample_bytree": {"low": 0.5, "high": 1.0},
                "min_child_weight": {"low": 1, "high": 10},
                "reg_alpha": {"low": 0.0, "high": 1.0},
                "reg_lambda": {"low": 0.0, "high": 1.0},
            },
            "tuning_trials": 5,
            "early_stopping_rounds": 10,
        },
        "ensemble": {
            "strategy": "weighted_average",
            "weights": {"lightgbm": 0.5, "xgboost": 0.5},
            "calibration": {"enabled": True, "method": "isotonic"},
        },
        "evaluation": {
            "primary_metric": "roc_auc",
            "k_values": [5, 10],
        },
        "shap": {"max_samples": 50},
    }


# ---------------------------------------------------------------------------
# TestLoadFold
# ---------------------------------------------------------------------------

class TestLoadFold:
    """Tests for data loading from fold parquets."""

    def test_load_fold_shapes(self, tmp_path):
        """Loaded arrays have correct shapes."""
        from src.models.trainer import load_fold

        features = _make_fold_parquets(tmp_path, fold_num=1, n_train=100, n_val=30, n_test=30)

        X_train, y_train, X_val, y_val, X_test, y_test, meta, feat_names = load_fold(tmp_path, 1)

        assert X_train.shape == (100, len(features))
        assert X_val.shape == (30, len(features))
        assert X_test.shape == (30, len(features))
        assert y_train.shape == (100,)
        assert y_val.shape == (30,)
        assert y_test.shape == (30,)
        assert len(feat_names) == len(features)

    def test_feature_columns_exclude_protected(self, tmp_path):
        """Feature columns should not include metadata columns."""
        from src.models.trainer import get_feature_columns

        df, _ = _make_synthetic_df(50)
        feat_cols = get_feature_columns(df)

        for col in ["player_id", "name", "team", "league", "season", "label",
                     "position", "position_group"]:
            # player_id is string so won't be in numeric cols anyway
            if col in df.select_dtypes(include=["number"]).columns.tolist():
                assert col not in feat_cols

        # All feature columns should be present
        for i in range(10):
            assert f"feat_{i}" in feat_cols

    def test_nan_filled_with_median(self, tmp_path):
        """NaN values should be filled with training column median."""
        from src.models.trainer import load_fold

        # Create data with NaN
        df, features = _make_synthetic_df(100, seed=42)
        # Inject NaNs
        df.loc[0, "feat_0"] = np.nan
        df.loc[1, "feat_0"] = np.nan

        df.to_parquet(tmp_path / "fold_1_train.parquet", index=False)
        df.to_parquet(tmp_path / "fold_1_val.parquet", index=False)
        df.to_parquet(tmp_path / "fold_1_test.parquet", index=False)

        X_train, _, _, _, _, _, _, _ = load_fold(tmp_path, 1)

        # No NaN should remain
        assert not np.any(np.isnan(X_train))

    def test_meta_test_has_metadata(self, tmp_path):
        """meta_test should contain protected columns from test set."""
        from src.models.trainer import load_fold

        _make_fold_parquets(tmp_path, fold_num=1, n_test=30)
        _, _, _, _, _, _, meta, _ = load_fold(tmp_path, 1)

        assert "player_id" in meta.columns
        assert "name" in meta.columns
        assert len(meta) == 30


# ---------------------------------------------------------------------------
# TestTrainer
# ---------------------------------------------------------------------------

class TestTrainer:
    """Tests for model training functions."""

    @pytest.fixture
    def training_data(self):
        """Generate training/validation arrays."""
        train_df, features = _make_synthetic_df(200, n_features=10, positive_rate=0.1, seed=42)
        val_df, _ = _make_synthetic_df(50, n_features=10, positive_rate=0.1, seed=43)

        from src.models.trainer import get_feature_columns
        feat_cols = get_feature_columns(train_df)

        X_train = train_df[feat_cols].values.astype(np.float64)
        y_train = train_df["label"].values.astype(np.float64)
        X_val = val_df[feat_cols].values.astype(np.float64)
        y_val = val_df["label"].values.astype(np.float64)

        return X_train, y_train, X_val, y_val

    def test_baseline_lr_trains(self, training_data):
        """Logistic regression trains and predicts probabilities in [0, 1]."""
        from src.models.trainer import train_baseline

        X_train, y_train, X_val, y_val = training_data
        config = _get_config()

        model, scaler = train_baseline(X_train, y_train, config)

        # Predict on validation
        X_val_scaled = scaler.transform(X_val)
        proba = model.predict_proba(X_val_scaled)[:, 1]

        assert proba.shape == (len(X_val),)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_lgbm_trains_with_early_stopping(self, training_data):
        """LightGBM trains and returns a Booster."""
        import lightgbm as lgb_lib
        from src.models.trainer import train_lgbm

        X_train, y_train, X_val, y_val = training_data
        config = _get_config()
        params = config["lightgbm"]["base_params"].copy()

        model = train_lgbm(X_train, y_train, X_val, y_val, params, config)

        assert isinstance(model, lgb_lib.Booster)
        # Early stopping means it stopped before 1000
        assert model.current_iteration() <= 1000

        proba = model.predict(X_val)
        assert proba.shape == (len(X_val),)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_xgb_trains_with_early_stopping(self, training_data):
        """XGBoost trains and returns a Booster."""
        import xgboost as xgb_lib
        from src.models.trainer import train_xgb

        X_train, y_train, X_val, y_val = training_data
        config = _get_config()
        params = config["xgboost"]["base_params"].copy()

        model = train_xgb(X_train, y_train, X_val, y_val, params, config)

        assert isinstance(model, xgb_lib.Booster)

        dval = xgb_lib.DMatrix(X_val)
        proba = model.predict(dval)
        assert proba.shape == (len(X_val),)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_class_weight_applied(self, training_data):
        """Models should use scale_pos_weight from config."""
        import lightgbm as lgb_lib
        from src.models.trainer import train_lgbm

        X_train, y_train, X_val, y_val = training_data
        config = _get_config()
        config["imbalance"]["positive_weight"] = 20

        params = config["lightgbm"]["base_params"].copy()
        model = train_lgbm(X_train, y_train, X_val, y_val, params, config)

        # The model should have the param set
        model_params = model.params
        assert model_params.get("scale_pos_weight") == 20


# ---------------------------------------------------------------------------
# TestEvaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    """Tests for evaluation metrics, ensemble, and calibration."""

    def test_precision_at_k_known_values(self):
        """Precision@K with known arrangement."""
        from src.models.evaluator import precision_at_k

        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

        # Top 5: first 5 predictions, 3 are positive
        assert precision_at_k(y_true, y_proba, 5) == pytest.approx(3 / 5)
        # Top 3: all positive
        assert precision_at_k(y_true, y_proba, 3) == pytest.approx(1.0)
        # Top 10: 3 positives
        assert precision_at_k(y_true, y_proba, 10) == pytest.approx(3 / 10)

    def test_recall_at_k_known_values(self):
        """Recall@K with known arrangement."""
        from src.models.evaluator import recall_at_k

        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

        # Top 2: 2 out of 3 positives found
        assert recall_at_k(y_true, y_proba, 2) == pytest.approx(2 / 3)
        # Top 3: all 3 positives found
        assert recall_at_k(y_true, y_proba, 3) == pytest.approx(1.0)

    def test_roc_auc_perfect(self):
        """ROC-AUC = 1.0 for perfect separation."""
        from src.models.evaluator import compute_metrics

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])

        metrics = compute_metrics(y_true, y_proba, k_values=[5])
        assert metrics["roc_auc"] == pytest.approx(1.0)

    def test_brier_score_perfect(self):
        """Brier score = 0 for perfectly calibrated predictions."""
        from src.models.evaluator import compute_metrics

        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])

        metrics = compute_metrics(y_true, y_proba, k_values=[2])
        assert metrics["brier_score"] == pytest.approx(0.0)

    def test_ensemble_weighted_average(self):
        """Ensemble is correct weighted average."""
        from src.models.evaluator import ensemble_predictions

        lgbm = np.array([0.2, 0.8, 0.5])
        xgb_ = np.array([0.4, 0.6, 0.3])

        result = ensemble_predictions(lgbm, xgb_, weights=(0.6, 0.4))
        expected = 0.6 * lgbm + 0.4 * xgb_
        np.testing.assert_array_almost_equal(result, expected)

    def test_ensemble_equal_weights(self):
        """Ensemble with equal weights is simple average."""
        from src.models.evaluator import ensemble_predictions

        lgbm = np.array([0.2, 0.8])
        xgb_ = np.array([0.4, 0.6])

        result = ensemble_predictions(lgbm, xgb_)
        np.testing.assert_array_almost_equal(result, [0.3, 0.7])

    def test_calibration_isotonic(self):
        """Isotonic calibration produces valid probabilities."""
        from src.models.evaluator import calibrate_probabilities

        rng = np.random.RandomState(42)
        y_val = (rng.rand(100) > 0.9).astype(float)
        proba_val = rng.rand(100)
        proba_test = rng.rand(50)

        calibrator, calibrated = calibrate_probabilities(y_val, proba_val, proba_test)

        assert calibrated.shape == (50,)
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_cross_fold_summary(self):
        """Cross-fold summary computes mean and std."""
        from src.models.evaluator import cross_fold_summary

        fold1 = {"lgbm_metrics": {"roc_auc": 0.8}, "xgb_metrics": {"roc_auc": 0.75},
                  "ensemble_metrics": {"roc_auc": 0.82}, "calibrated_metrics": {"roc_auc": 0.81}}
        fold2 = {"lgbm_metrics": {"roc_auc": 0.85}, "xgb_metrics": {"roc_auc": 0.78},
                  "ensemble_metrics": {"roc_auc": 0.86}, "calibrated_metrics": {"roc_auc": 0.84}}

        summary = cross_fold_summary([fold1, fold2])

        assert "lgbm_metrics" in summary
        assert summary["lgbm_metrics"]["roc_auc"]["mean"] == pytest.approx(0.825)
        assert summary["lgbm_metrics"]["roc_auc"]["std"] == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# TestTuner
# ---------------------------------------------------------------------------

class TestTuner:
    """Tests for Optuna hyperparameter tuning."""

    @pytest.fixture
    def tuning_data(self):
        """Generate data for tuning tests (needs separable signal)."""
        rng = np.random.RandomState(42)
        n_train, n_val = 200, 50
        n_features = 10

        X_train = rng.randn(n_train, n_features)
        # Make labels correlated with first feature for signal
        y_train = (X_train[:, 0] > 0).astype(float)

        X_val = rng.randn(n_val, n_features)
        y_val = (X_val[:, 0] > 0).astype(float)

        return X_train, y_train, X_val, y_val

    def test_tune_lgbm_returns_params(self, tuning_data):
        """Optuna returns valid LightGBM params."""
        from src.models.tuner import tune_lgbm

        X_train, y_train, X_val, y_val = tuning_data
        config = _get_config()

        best_params, study = tune_lgbm(X_train, y_train, X_val, y_val, config, n_trials=3)

        assert isinstance(best_params, dict)
        assert "num_leaves" in best_params
        assert "learning_rate" in best_params
        assert study.best_value > 0  # ROC-AUC should be positive

    def test_tune_xgb_returns_params(self, tuning_data):
        """Optuna returns valid XGBoost params."""
        from src.models.tuner import tune_xgb

        X_train, y_train, X_val, y_val = tuning_data
        config = _get_config()

        best_params, study = tune_xgb(X_train, y_train, X_val, y_val, config, n_trials=3)

        assert isinstance(best_params, dict)
        assert "max_depth" in best_params
        assert "learning_rate" in best_params
        assert study.best_value > 0

    def test_tuner_roc_auc_in_range(self, tuning_data):
        """ROC-AUC objective returns float in [0, 1]."""
        from src.models.tuner import tune_lgbm

        X_train, y_train, X_val, y_val = tuning_data
        config = _get_config()

        _, study = tune_lgbm(X_train, y_train, X_val, y_val, config, n_trials=3)

        for trial in study.trials:
            assert 0.0 <= trial.value <= 1.0


# ---------------------------------------------------------------------------
# TestExplainer
# ---------------------------------------------------------------------------

class TestExplainer:
    """Tests for SHAP explanations."""

    @pytest.fixture
    def trained_models(self):
        """Train small LGBM and XGB models for SHAP tests."""
        import lightgbm as lgb_lib
        import xgboost as xgb_lib
        from src.models.trainer import train_lgbm, train_xgb

        rng = np.random.RandomState(42)
        n_train, n_val, n_test = 200, 50, 50
        n_features = 8
        feature_names = [f"feat_{i}" for i in range(n_features)]

        X_train = rng.randn(n_train, n_features)
        y_train = (X_train[:, 0] > 0).astype(float)
        X_val = rng.randn(n_val, n_features)
        y_val = (X_val[:, 0] > 0).astype(float)
        X_test = rng.randn(n_test, n_features)
        y_test = (X_test[:, 0] > 0).astype(float)

        config = _get_config()
        lgbm_params = {"num_leaves": 15, "learning_rate": 0.1, "verbose": -1}
        xgb_params = {"max_depth": 4, "learning_rate": 0.1, "verbosity": 0}

        lgbm_model = train_lgbm(X_train, y_train, X_val, y_val, lgbm_params, config)
        xgb_model = train_xgb(X_train, y_train, X_val, y_val, xgb_params, config)

        meta = pd.DataFrame({
            "player_id": [f"p{i}" for i in range(n_test)],
            "name": [f"Player {i}" for i in range(n_test)],
        })

        return lgbm_model, xgb_model, X_test, y_test, feature_names, meta

    def test_shap_values_shape(self, trained_models):
        """SHAP values have correct shape (n_samples, n_features)."""
        from src.models.explainer import compute_shap_values

        lgbm_model, _, X_test, _, feature_names, _ = trained_models

        shap_vals = compute_shap_values(lgbm_model, X_test, feature_names, max_samples=30)

        # Should have shape (n_samples, n_features) — possibly 3D for multi-output
        assert shap_vals.values.shape[0] <= 30
        if shap_vals.values.ndim == 2:
            assert shap_vals.values.shape[1] == len(feature_names)
        else:
            assert shap_vals.values.shape[1] == len(feature_names)

    def test_feature_importance_sorted(self, trained_models):
        """Feature importance returns sorted DataFrame."""
        from src.models.explainer import compute_shap_values, feature_importance

        lgbm_model, _, X_test, _, feature_names, _ = trained_models

        shap_vals = compute_shap_values(lgbm_model, X_test, feature_names, max_samples=30)
        importance_df = feature_importance(shap_vals, feature_names)

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == len(feature_names)
        # Should be sorted descending
        assert importance_df["importance"].is_monotonic_decreasing

    def test_explain_player(self, trained_models):
        """Player explanation contains expected keys."""
        from src.models.explainer import compute_shap_values, explain_player

        lgbm_model, _, X_test, _, feature_names, meta = trained_models

        shap_vals = compute_shap_values(lgbm_model, X_test, feature_names, max_samples=30)
        explanation = explain_player(shap_vals, 0, feature_names, meta, top_n=5)

        assert "top_features" in explanation
        assert len(explanation["top_features"]) == 5
        assert "base_value" in explanation
        assert "player_id" in explanation
        assert explanation["top_features"][0]["feature"] in feature_names

    def test_generate_explanations(self, trained_models):
        """Full explanation pipeline produces combined importance."""
        from src.models.explainer import generate_explanations

        lgbm_model, xgb_model, X_test, _, feature_names, meta = trained_models

        result = generate_explanations(
            lgbm_model, xgb_model, X_test, feature_names, meta, max_samples=30
        )

        assert "lgbm_shap" in result
        assert "xgb_shap" in result
        assert "combined_importance" in result
        assert len(result["combined_importance"]) == len(feature_names)


# ---------------------------------------------------------------------------
# Integration-style test: evaluate_fold
# ---------------------------------------------------------------------------

class TestEvaluateFold:
    """Integration test for the full evaluation pipeline on one fold."""

    def test_evaluate_fold_end_to_end(self):
        """evaluate_fold produces metrics and predictions DataFrame."""
        import xgboost as xgb_lib
        from src.models.trainer import train_lgbm, train_xgb
        from src.models.evaluator import evaluate_fold

        rng = np.random.RandomState(42)
        n_train, n_val, n_test = 200, 60, 60
        n_features = 8

        X_train = rng.randn(n_train, n_features)
        y_train = (X_train[:, 0] > 0).astype(float)
        X_val = rng.randn(n_val, n_features)
        y_val = (X_val[:, 0] > 0).astype(float)
        X_test = rng.randn(n_test, n_features)
        y_test = (X_test[:, 0] > 0).astype(float)

        meta_test = pd.DataFrame({
            "player_id": [f"p{i}" for i in range(n_test)],
            "name": [f"Player {i}" for i in range(n_test)],
            "season": ["2020-21"] * n_test,
        })

        config = _get_config()
        lgbm_model = train_lgbm(
            X_train, y_train, X_val, y_val,
            {"num_leaves": 15, "learning_rate": 0.1, "verbose": -1}, config
        )
        xgb_model = train_xgb(
            X_train, y_train, X_val, y_val,
            {"max_depth": 4, "learning_rate": 0.1, "verbosity": 0}, config
        )

        result = evaluate_fold(
            lgbm_model, xgb_model, X_test, y_test, X_val, y_val, meta_test, config
        )

        # Check structure
        assert "lgbm_metrics" in result
        assert "calibrated_metrics" in result
        assert "predictions" in result

        # Metrics should be valid
        assert 0 <= result["calibrated_metrics"]["roc_auc"] <= 1
        assert result["calibrated_metrics"]["brier_score"] >= 0

        # Predictions DataFrame
        preds = result["predictions"]
        assert "prob_calibrated" in preds.columns
        assert "player_id" in preds.columns
        assert len(preds) == n_test
        assert preds["prob_calibrated"].between(0, 1).all()
