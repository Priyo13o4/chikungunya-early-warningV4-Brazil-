"""Baseline model definitions and production-safe training interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from src.models.baselines.threshold_rules import DistrictPercentileThresholdRule

LOGGER = logging.getLogger(__name__)

MODEL_ALIASES: dict[str, str] = {
    "rule": "rule_based_threshold",
    "rule_based": "rule_based_threshold",
    "rule_based_threshold": "rule_based_threshold",
    "logit": "logistic_regression",
    "logistic": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "poisson": "poisson_regression",
    "poisson_regression": "poisson_regression",
    "negative_binomial": "negative_binomial_regression",
    "neg_bin": "negative_binomial_regression",
    "negative_binomial_regression": "negative_binomial_regression",
    "rf": "random_forest",
    "random_forest": "random_forest",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "lgbm": "lightgbm",
    "lightgbm": "lightgbm",
}


def _to_numeric_target(y: pd.Series) -> pd.Series:
    values = pd.to_numeric(y, errors="coerce").fillna(0.0)
    return values.astype(float)


def _to_float_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame.copy()
    if matrix.empty:
        return matrix
    for column in matrix.columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").fillna(0.0)
    return matrix.astype(float)


@dataclass
class BaselineModel:
    """Unified baseline model wrapper with robust sparse-data behavior."""

    name: str
    random_state: int = 42
    compute_backend: str = "cpu"
    district_column: str = "district"
    case_column: str = "cases"
    estimator_: Any = None
    encoded_columns_: list[str] = field(default_factory=list)
    feature_name_map_: dict[str, str] = field(default_factory=dict)
    feature_name_inverse_map_: dict[str, str] = field(default_factory=dict)
    target_mean_: float = 0.0
    fitted_: bool = False

    def _resolve_case_signal_column(self, frame: pd.DataFrame) -> str | None:
        candidates = [
            self.case_column,
            "case_lag_1",
            "cases_lag_1",
            "case_signal",
        ]
        for column in candidates:
            if column in frame.columns:
                return column
        return None

    def _prepare_rule_frame(self, frame: pd.DataFrame) -> pd.DataFrame | None:
        case_source = self._resolve_case_signal_column(frame)
        if case_source is None:
            LOGGER.warning(
                "Model '%s' missing case signal columns (%s); using constant fallback",
                self.name,
                ", ".join([self.case_column, "case_lag_1", "cases_lag_1", "case_signal"]),
            )
            return None

        if case_source == self.case_column:
            prepared = frame.copy()
        else:
            prepared = frame.copy()
            prepared[self.case_column] = pd.to_numeric(prepared[case_source], errors="coerce").fillna(0.0)

        if self.district_column not in prepared.columns:
            prepared[self.district_column] = "__all__"

        return prepared

    def _normalize_name(self) -> str:
        normalized = MODEL_ALIASES.get(self.name.lower(), self.name.lower())
        return normalized

    def _encode_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """One-hot encode non-numeric columns with train/test column alignment."""
        frame = X.copy()
        if frame.empty:
            encoded = pd.DataFrame(index=frame.index)
        else:
            encoded = pd.get_dummies(frame, dummy_na=True)
            for column in encoded.columns:
                encoded[column] = pd.to_numeric(encoded[column], errors="coerce").fillna(0.0)
            encoded = encoded.reindex(sorted(encoded.columns), axis=1)
        if fit:
            self.encoded_columns_ = list(encoded.columns)
            return encoded

        if not self.encoded_columns_:
            return encoded
        return encoded.reindex(columns=self.encoded_columns_, fill_value=0.0)

    def _uses_booster_name_constraints(self) -> bool:
        return self._normalize_name() in {"xgboost", "lightgbm"}

    def _normalized_compute_backend(self) -> str:
        return str(self.compute_backend or "cpu").strip().lower()

    def _xgboost_backend_overrides(self) -> dict[str, Any]:
        backend = self._normalized_compute_backend()
        if backend == "nvidia_cuda":
            return {
                "device": "cuda",
                "tree_method": "hist",
            }
        if backend == "macos_metal":
            LOGGER.warning("Model '%s': xgboost does not support macOS Metal directly; using CPU", self.name)
            return {}
        if backend != "cpu":
            LOGGER.warning("Model '%s': unknown compute backend '%s'; using CPU", self.name, backend)
        return {}

    def _lightgbm_backend_overrides(self) -> dict[str, Any]:
        backend = self._normalized_compute_backend()
        if backend == "nvidia_cuda":
            return {"device_type": "gpu"}
        if backend == "macos_metal":
            LOGGER.warning("Model '%s': lightgbm does not support macOS Metal directly; using CPU", self.name)
            return {}
        if backend != "cpu":
            LOGGER.warning("Model '%s': unknown compute backend '%s'; using CPU", self.name, backend)
        return {}

    @staticmethod
    def _sanitize_column_name(column_name: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z_]", "_", str(column_name))
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        if not sanitized:
            return "feature"
        if sanitized[0].isdigit():
            sanitized = f"f_{sanitized}"
        return sanitized

    def _apply_feature_name_sanitization(self, matrix: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if not self._uses_booster_name_constraints():
            return matrix

        if fit:
            used_names: set[str] = set()
            mapping: dict[str, str] = {}
            inverse_mapping: dict[str, str] = {}
            for original_name in matrix.columns:
                base_name = self._sanitize_column_name(original_name)
                sanitized_name = base_name
                suffix = 1
                while sanitized_name in used_names:
                    suffix += 1
                    sanitized_name = f"{base_name}_{suffix}"
                used_names.add(sanitized_name)
                mapping[str(original_name)] = sanitized_name
                inverse_mapping[sanitized_name] = str(original_name)

            self.feature_name_map_ = mapping
            self.feature_name_inverse_map_ = inverse_mapping

        if not self.feature_name_map_:
            return matrix

        rename_mapping = {
            column: self.feature_name_map_.get(str(column), self._sanitize_column_name(str(column)))
            for column in matrix.columns
        }
        return matrix.rename(columns=rename_mapping)

    def _set_constant_fallback(self, y: pd.Series, reason: str) -> None:
        y_numeric = _to_numeric_target(y)
        self.target_mean_ = float(y_numeric.mean()) if len(y_numeric) else 0.0
        self.estimator_ = "constant"
        LOGGER.warning("Model '%s' using constant fallback: %s", self.name, reason)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineModel":
        """Fit baseline model with graceful degradation when dependencies are missing."""
        model_name = self._normalize_name()
        y_numeric = _to_numeric_target(y)
        self.target_mean_ = float(y_numeric.mean()) if len(y_numeric) else 0.0

        if y_numeric.nunique() <= 1:
            self._set_constant_fallback(y_numeric, "single-class training labels")
            self.fitted_ = True
            return self

        if model_name == "rule_based_threshold":
            rule_frame = self._prepare_rule_frame(X)
            if rule_frame is None:
                self._set_constant_fallback(y_numeric, "missing case signal for threshold rule")
                self.fitted_ = True
                return self

            rule = DistrictPercentileThresholdRule(
                percentile=0.75,
                district_column=self.district_column,
                case_column=self.case_column,
            )
            rule.fit(rule_frame)
            self.estimator_ = rule
            self.fitted_ = True
            return self

        matrix = self._encode_features(X, fit=True)
        matrix = self._apply_feature_name_sanitization(matrix, fit=True)
        matrix = _to_float_matrix(matrix)

        try:
            if model_name == "logistic_regression":
                from sklearn.linear_model import LogisticRegression

                estimator = LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=self.random_state,
                    solver="lbfgs",
                )
                estimator.fit(matrix, y_numeric)
                self.estimator_ = estimator

            elif model_name == "poisson_regression":
                from sklearn.linear_model import PoissonRegressor

                estimator = PoissonRegressor(alpha=1e-3, max_iter=500)
                estimator.fit(matrix, y_numeric)
                self.estimator_ = estimator

            elif model_name == "negative_binomial_regression":
                try:
                    import statsmodels.api as sm

                    design = sm.add_constant(matrix, has_constant="add")
                    design = _to_float_matrix(design)
                    design = design.reset_index(drop=True)
                    endog = _to_numeric_target(y_numeric).astype(float).reset_index(drop=True)
                    estimator = sm.GLM(
                        endog,
                        design,
                        family=sm.families.NegativeBinomial(alpha=1.0),
                    ).fit(maxiter=200, disp=0)
                    self.estimator_ = estimator
                except Exception as nb_error:
                    LOGGER.warning(
                        "Negative-binomial fit failed for '%s'; falling back to Poisson (%s)",
                        self.name,
                        nb_error,
                    )
                    from sklearn.linear_model import PoissonRegressor

                    estimator = PoissonRegressor(alpha=1e-3, max_iter=500)
                    estimator.fit(matrix, y_numeric)
                    self.estimator_ = estimator

            elif model_name == "random_forest":
                from sklearn.ensemble import RandomForestClassifier

                estimator = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=self.random_state,
                )
                estimator.fit(matrix, y_numeric.astype(int))
                self.estimator_ = estimator

            elif model_name == "xgboost":
                try:
                    from xgboost import XGBClassifier
                except Exception as dependency_error:
                    self._set_constant_fallback(y_numeric, f"xgboost unavailable ({dependency_error})")
                    self.fitted_ = True
                    return self

                base_kwargs = {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "max_depth": 4,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "random_state": self.random_state,
                    "n_jobs": 1,
                }
                backend_overrides = self._xgboost_backend_overrides()
                estimator = XGBClassifier(
                    **{
                        **base_kwargs,
                        **backend_overrides,
                    }
                )
                try:
                    estimator.fit(matrix, y_numeric.astype(int))
                except Exception as fit_error:
                    if backend_overrides:
                        LOGGER.warning(
                            "Model '%s': xgboost backend '%s' unavailable (%s); retrying on CPU",
                            self.name,
                            self._normalized_compute_backend(),
                            fit_error,
                        )
                        estimator = XGBClassifier(**base_kwargs)
                        estimator.fit(matrix, y_numeric.astype(int))
                    else:
                        raise
                self.estimator_ = estimator

            elif model_name == "lightgbm":
                try:
                    from lightgbm import LGBMClassifier
                except Exception as dependency_error:
                    self._set_constant_fallback(y_numeric, f"lightgbm unavailable ({dependency_error})")
                    self.fitted_ = True
                    return self

                base_kwargs = {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "num_leaves": 31,
                    "force_row_wise": True,
                    "verbosity": -1,
                    "random_state": self.random_state,
                }
                backend_overrides = self._lightgbm_backend_overrides()
                estimator = LGBMClassifier(
                    **{
                        **base_kwargs,
                        **backend_overrides,
                    }
                )
                try:
                    estimator.fit(matrix, y_numeric.astype(int))
                except Exception as fit_error:
                    if backend_overrides:
                        LOGGER.warning(
                            "Model '%s': lightgbm backend '%s' unavailable (%s); retrying on CPU",
                            self.name,
                            self._normalized_compute_backend(),
                            fit_error,
                        )
                        estimator = LGBMClassifier(**base_kwargs)
                        estimator.fit(matrix, y_numeric.astype(int))
                    else:
                        raise
                self.estimator_ = estimator

            else:
                self._set_constant_fallback(y_numeric, f"unknown model '{self.name}'")

        except Exception as fit_error:
            self._set_constant_fallback(y_numeric, str(fit_error))

        self.fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return outbreak probabilities in [0, 1]."""
        if not self.fitted_:
            raise RuntimeError(f"Model '{self.name}' must be fit before prediction")

        if isinstance(self.estimator_, DistrictPercentileThresholdRule):
            rule_frame = self._prepare_rule_frame(X)
            if rule_frame is None:
                return pd.Series(self.target_mean_, index=X.index, name=f"{self.name}_proba").clip(0.0, 1.0)
            predictions = self.estimator_.predict_proba(rule_frame)
            return predictions.reindex(X.index).fillna(self.target_mean_).clip(0.0, 1.0)

        if self.estimator_ == "constant" or self.estimator_ is None:
            return pd.Series(self.target_mean_, index=X.index, name=f"{self.name}_proba").clip(0.0, 1.0)

        matrix = self._encode_features(X, fit=False)
        matrix = self._apply_feature_name_sanitization(matrix, fit=False)
        matrix = _to_float_matrix(matrix)
        try:
            if hasattr(self.estimator_, "predict_proba"):
                values = self.estimator_.predict_proba(matrix)
                if isinstance(values, np.ndarray) and values.ndim == 2 and values.shape[1] > 1:
                    proba = values[:, 1]
                else:
                    proba = np.asarray(values).reshape(-1)
            elif hasattr(self.estimator_, "model") and hasattr(self.estimator_.model, "exog_names"):
                try:
                    import statsmodels.api as sm

                    design = sm.add_constant(matrix, has_constant="add")
                    design = _to_float_matrix(design)
                    exog_names = [str(name) for name in getattr(self.estimator_.model, "exog_names", [])]
                    if exog_names:
                        design = design.reindex(columns=exog_names, fill_value=0.0)
                    preds = self.estimator_.predict(design)
                    proba = np.asarray(preds).reshape(-1)
                except Exception:
                    preds = self.estimator_.predict(matrix)
                    proba = np.asarray(preds).reshape(-1)
            else:
                preds = self.estimator_.predict(matrix)
                proba = np.asarray(preds).reshape(-1)
                proba = 1.0 - np.exp(-np.clip(proba, a_min=0.0, a_max=None))
        except Exception as predict_error:
            LOGGER.warning(
                "Prediction failed for model '%s'; using mean fallback (%s)",
                self.name,
                predict_error,
            )
            proba = np.full(shape=len(X), fill_value=self.target_mean_, dtype=float)

        series = pd.Series(proba, index=X.index, name=f"{self.name}_proba", dtype="float64")
        return series.clip(lower=0.0, upper=1.0)


def get_baseline_model(name: str, *, random_state: int = 42, compute_backend: str = "cpu") -> BaselineModel:
    """Return a baseline model instance by canonical model name or alias."""
    return BaselineModel(
        name=MODEL_ALIASES.get(name.lower(), name.lower()),
        random_state=random_state,
        compute_backend=compute_backend,
    )
