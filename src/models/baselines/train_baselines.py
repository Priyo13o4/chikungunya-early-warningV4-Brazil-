"""Training entry points for baseline models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import pickle
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from config.paths import ensure_directories
from src.models.baselines.baseline_models import BaselineModel, get_baseline_model
from src.models.baselines.calibrate_baselines import brier_score
from src.models.baselines.cv_splitter import TimeSeriesCVConfig, build_fold_ledger, generate_time_splits
from src.models.baselines.model_registry import list_default_model_names, validate_model_names

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Configuration for temporal-CV baseline model training."""

    date_column: str = "date"
    output_subdir: str = "baselines"
    random_state: int = 42
    compute_backend: str = "cpu"
    enable_temporal_cv: bool = True
    calibration_method: str = "isotonic"


def _accuracy(y_true: pd.Series, y_prob: pd.Series, threshold: float = 0.5) -> float:
    labels = pd.to_numeric(y_true, errors="coerce").fillna(0.0).astype(int)
    preds = (pd.to_numeric(y_prob, errors="coerce").fillna(0.0) >= threshold).astype(int)
    if len(labels) == 0:
        return 0.0
    return float((labels == preds).mean())


def _ensure_output_dir(output_dir: Path | None, config: BaselineTrainingConfig) -> Path:
    if output_dir is not None:
        target = output_dir
    else:
        paths = ensure_directories()
        target = paths.outputs_models / config.output_subdir
    target.mkdir(parents=True, exist_ok=True)
    return target


def _save_pickle(model: BaselineModel, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as file_obj:
        pickle.dump(model, file_obj)


def _train_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
    compute_backend: str,
) -> BaselineModel:
    model = get_baseline_model(model_name, random_state=random_state, compute_backend=compute_backend)
    return model.fit(X_train, y_train)


def train_baselines(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: Iterable[str] | None = None,
    *,
    config: BaselineTrainingConfig = BaselineTrainingConfig(),
    cv_config: TimeSeriesCVConfig | None = None,
    output_dir: Path | None = None,
    build_fold_ledger_fn: Callable[[pd.DataFrame, TimeSeriesCVConfig], list[dict[str, Any]]] = build_fold_ledger,
    generate_time_splits_fn: Callable[[pd.DataFrame, TimeSeriesCVConfig], Any] = generate_time_splits,
) -> dict[str, BaselineModel]:
    """Train baseline models with leakage-safe temporal CV and persistence.

    Returns final models trained on all rows.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have same length")

    requested_names = list(model_names) if model_names is not None else list_default_model_names()
    selected_names = validate_model_names(requested_names)
    if not selected_names:
        LOGGER.warning("No valid requested models found; falling back to default baseline suite")
        selected_names = list_default_model_names()

    output_root = _ensure_output_dir(output_dir, config=config)
    cv_cfg = cv_config or TimeSeriesCVConfig(date_column=config.date_column)

    training_frame = X.copy()
    training_frame[cv_cfg.target_column] = pd.to_numeric(y, errors="coerce").fillna(0.0)

    cv_metrics_records: list[dict[str, float | int | str]] = []
    fold_ledger = build_fold_ledger_fn(training_frame, cv_cfg)
    (output_root / "fold_ledger.json").write_text(json.dumps(fold_ledger, indent=2), encoding="utf-8")
    if config.enable_temporal_cv:
        for fold_id, (train_idx, valid_idx) in enumerate(generate_time_splits_fn(training_frame, cv_cfg), start=1):
            fold_dir = output_root / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_valid = X.loc[valid_idx]
            y_valid = y.loc[valid_idx]

            if pd.to_numeric(y_train, errors="coerce").dropna().nunique() <= 1:
                LOGGER.info("Skipping fold_%d because training labels are single-class", fold_id)
                continue

            fold_predictions: dict[str, pd.Series] = {}
            for model_name in selected_names:
                try:
                    model = _train_single_model(
                        model_name=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        random_state=config.random_state,
                        compute_backend=config.compute_backend,
                    )
                    y_prob = model.predict_proba(X_valid)
                except Exception as model_error:
                    LOGGER.warning("Skipping model '%s' on fold_%d due to error: %s", model_name, fold_id, model_error)
                    continue
                fold_predictions[model_name] = y_prob

                _save_pickle(model, fold_dir / f"{model_name}.pkl")
                cv_metrics_records.append(
                    {
                        "fold": fold_id,
                        "model": model_name,
                        "rows_train": int(len(train_idx)),
                        "rows_valid": int(len(valid_idx)),
                        "accuracy": _accuracy(y_valid, y_prob),
                        "brier": brier_score(y_valid, y_prob),
                        "positive_rate": float(pd.to_numeric(y_prob, errors="coerce").fillna(0.0).mean()),
                    }
                )

            if fold_predictions:
                pd.DataFrame(fold_predictions, index=valid_idx).to_csv(
                    fold_dir / "predictions.csv",
                    index=True,
                )

    final_dir = output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trained_models: dict[str, BaselineModel] = {}
    final_predictions: dict[str, pd.Series] = {}

    for model_name in selected_names:
        try:
            model = _train_single_model(
                model_name=model_name,
                X_train=X,
                y_train=y,
                random_state=config.random_state,
                compute_backend=config.compute_backend,
            )
        except Exception as model_error:
            LOGGER.warning("Skipping final training for model '%s' due to error: %s", model_name, model_error)
            continue
        trained_models[model_name] = model
        final_predictions[model_name] = model.predict_proba(X)
        _save_pickle(model, final_dir / f"{model_name}.pkl")

    if final_predictions:
        pd.DataFrame(final_predictions, index=X.index).to_csv(final_dir / "predictions.csv", index=True)

    metrics_df = pd.DataFrame(cv_metrics_records)
    metrics_df.to_csv(output_root / "cv_metrics.csv", index=False)

    aggregate_metrics = (
        metrics_df.groupby("model", dropna=False)[["accuracy", "brier", "positive_rate"]].mean().reset_index()
        if not metrics_df.empty
        else pd.DataFrame(columns=["model", "accuracy", "brier", "positive_rate"])
    )
    aggregate_metrics.to_csv(output_root / "cv_metrics_aggregate.csv", index=False)

    with (output_root / "training_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "baseline_config": asdict(config),
                "cv_config": asdict(cv_cfg),
                "cv_fold_ledger_path": str(output_root / "fold_ledger.json"),
                "models": selected_names,
                "n_rows": int(len(X)),
                "target_positive_rate": float(np.mean(pd.to_numeric(y, errors="coerce").fillna(0.0))),
            },
            file_obj,
            indent=2,
        )

    LOGGER.info("Trained %d baseline models. Artifacts saved to %s", len(trained_models), output_root)
    return trained_models
