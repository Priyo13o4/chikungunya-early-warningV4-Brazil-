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
from src.models.baselines.cv_splitter import TimeSeriesCVConfig, build_fold_ledger, generate_time_splits
from src.models.baselines.model_registry import list_default_model_names, validate_model_names

LOGGER = logging.getLogger(__name__)

_CLIMATE_COLUMN_TOKENS: tuple[str, ...] = ("rain", "temp", "humid", "precip", "climate", "lai")


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
    labels = pd.to_numeric(y_true, errors="coerce").fillna(0.0).to_numpy(dtype=int)
    probs = pd.to_numeric(y_prob, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if labels.size == 0 or probs.size == 0:
        return 0.0
    n = min(labels.size, probs.size)
    if n == 0:
        return 0.0
    preds = (probs[:n] >= threshold).astype(int)
    return float((labels[:n] == preds).mean())


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


def _infer_climate_feature_columns(columns: Iterable[str]) -> list[str]:
    selected: list[str] = []
    for column in columns:
        lowered = str(column).lower()
        if any(token in lowered for token in _CLIMATE_COLUMN_TOKENS):
            selected.append(str(column))
    return sorted(set(selected))


def _impute_fold_climate_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    *,
    district_column: str = "district",
    month_column: str = "month",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    climate_columns = [column for column in _infer_climate_feature_columns(X_train.columns) if column in X_valid.columns]
    if not climate_columns:
        return X_train, X_valid

    train_output = X_train.copy()
    valid_output = X_valid.copy()
    has_group_keys = district_column in train_output.columns and month_column in train_output.columns
    if has_group_keys and month_column in valid_output.columns:
        train_month = pd.to_numeric(train_output[month_column], errors="coerce")
        valid_month = pd.to_numeric(valid_output[month_column], errors="coerce")

    for column in climate_columns:
        train_numeric = pd.to_numeric(train_output[column], errors="coerce")
        valid_numeric = pd.to_numeric(valid_output[column], errors="coerce")
        train_filled = train_numeric.copy()
        valid_filled = valid_numeric.copy()

        if has_group_keys and month_column in valid_output.columns:
            reference = pd.DataFrame(
                {
                    "district": train_output[district_column],
                    "month": train_month,
                    "value": train_numeric,
                }
            )
            district_month_median = reference.groupby(["district", "month"], dropna=False)["value"].median()

            train_keys = pd.MultiIndex.from_arrays([train_output[district_column], train_month])
            valid_keys = pd.MultiIndex.from_arrays([valid_output[district_column], valid_month])
            train_fill_values = pd.Series(train_keys.map(district_month_median), index=train_output.index)
            valid_fill_values = pd.Series(valid_keys.map(district_month_median), index=valid_output.index)
            train_filled = train_filled.fillna(train_fill_values)
            valid_filled = valid_filled.fillna(valid_fill_values)

        global_median = train_numeric.median(skipna=True)
        if pd.notna(global_median):
            train_filled = train_filled.fillna(float(global_median))
            valid_filled = valid_filled.fillna(float(global_median))

        train_output[column] = train_filled
        valid_output[column] = valid_filled

    return train_output, valid_output


def _build_future_case_series(
    *,
    case_series: pd.Series,
    district_series: pd.Series,
    temporal_series: pd.Series | None,
) -> pd.Series:
    working = pd.DataFrame(
        {
            "cases": pd.to_numeric(case_series, errors="coerce"),
            "district": district_series,
        },
        index=case_series.index,
    )
    if temporal_series is not None:
        working["date"] = pd.to_datetime(temporal_series, errors="coerce")
    else:
        working["date"] = pd.NaT

    working["_order"] = np.arange(len(working), dtype=int)
    working = working.sort_values(["district", "date", "_order"], na_position="last")
    working["future_cases"] = working.groupby("district", dropna=False)["cases"].shift(-1)
    restored = working.sort_values("_order")["future_cases"]
    restored.index = case_series.index
    return pd.to_numeric(restored, errors="coerce")


def _build_forward_horizon_position_series(
    *,
    district_series: pd.Series,
    temporal_series: pd.Series | None,
    horizon_steps: int,
) -> pd.Series:
    steps = max(1, int(horizon_steps))
    working = pd.DataFrame(
        {
            "district": district_series,
            "row_position": np.arange(len(district_series), dtype=int),
        },
        index=district_series.index,
    )
    if temporal_series is not None:
        working["date"] = pd.to_datetime(temporal_series, errors="coerce")
    else:
        working["date"] = pd.NaT

    working["_order"] = np.arange(len(working), dtype=int)
    working = working.sort_values(["district", "date", "_order"], na_position="last")
    working["forward_position"] = working.groupby("district", dropna=False)["row_position"].shift(-steps)
    restored = working.sort_values("_order")["forward_position"]
    restored.index = district_series.index
    return pd.to_numeric(restored, errors="coerce")


def _derive_fold_targets_from_train_threshold(
    *,
    case_series: pd.Series,
    district_series: pd.Series,
    future_case_series: pd.Series,
    train_positions: np.ndarray,
    valid_positions: np.ndarray,
    selected_percentile: int,
) -> tuple[pd.Series, pd.Series]:
    q = float(selected_percentile) / 100.0
    train_case = pd.to_numeric(case_series.iloc[train_positions], errors="coerce")
    train_district = district_series.iloc[train_positions]
    valid_district = district_series.iloc[valid_positions]

    train_threshold = train_case.groupby(train_district, dropna=False).quantile(q)
    train_threshold_values = train_district.map(train_threshold)
    valid_threshold_values = valid_district.map(train_threshold)

    train_future = pd.to_numeric(future_case_series.iloc[train_positions], errors="coerce")
    valid_future = pd.to_numeric(future_case_series.iloc[valid_positions], errors="coerce")

    y_train = (
        (train_future > pd.to_numeric(train_threshold_values, errors="coerce"))
        & train_future.notna()
        & pd.to_numeric(train_threshold_values, errors="coerce").notna()
    ).astype(int)
    y_valid = (
        (valid_future > pd.to_numeric(valid_threshold_values, errors="coerce"))
        & valid_future.notna()
        & pd.to_numeric(valid_threshold_values, errors="coerce").notna()
    ).astype(int)
    return y_train, y_valid


def train_baselines(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: Iterable[str] | None = None,
    *,
    config: BaselineTrainingConfig = BaselineTrainingConfig(),
    cv_config: TimeSeriesCVConfig | None = None,
    output_dir: Path | None = None,
    case_series: pd.Series | None = None,
    district_series: pd.Series | None = None,
    temporal_series: pd.Series | None = None,
    selected_percentile: int = 75,
    fold_local_labeling: bool = True,
    fold_local_climate_imputation: bool = True,
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
    if not bool(fold_local_climate_imputation):
        LOGGER.warning(
            "fold_local_climate_imputation_disabled_ignored | strict fold-local climate imputation remains enforced for baseline CV"
        )
    fold_local_climate_imputation = True

    training_frame = X.copy()
    training_frame[cv_cfg.target_column] = pd.to_numeric(y, errors="coerce").fillna(0.0)

    cv_metrics_records: list[dict[str, float | int | str]] = []
    fold_ledger = build_fold_ledger_fn(training_frame, cv_cfg)
    (output_root / "fold_ledger.json").write_text(json.dumps(fold_ledger, indent=2), encoding="utf-8")
    yielded_fold_ledger = [fold for fold in fold_ledger if str(fold.get("status")) == "yielded"]
    skipped_fold_ledger = [fold for fold in fold_ledger if str(fold.get("status")) != "yielded"]
    skipped_by_reason: dict[str, int] = {}
    for fold in skipped_fold_ledger:
        reason = str(fold.get("reason", "unknown"))
        skipped_by_reason[reason] = int(skipped_by_reason.get(reason, 0)) + 1
    LOGGER.info(
        "Baseline CV plan | total_folds=%d, yielded=%d, skipped=%d, minimum_evaluated_folds=%d",
        int(len(fold_ledger)),
        int(len(yielded_fold_ledger)),
        int(len(skipped_fold_ledger)),
        int(cv_cfg.minimum_evaluated_folds),
    )
    if skipped_by_reason:
        LOGGER.info("Baseline CV skipped fold reasons: %s", skipped_by_reason)
    case_ref = pd.Series(case_series.to_numpy(), index=X.index) if case_series is not None and len(case_series) == len(X) else None
    district_ref = pd.Series(district_series.to_numpy(), index=X.index) if district_series is not None and len(district_series) == len(X) else None
    temporal_ref = (
        pd.Series(temporal_series.to_numpy(), index=X.index)
        if temporal_series is not None and len(temporal_series) == len(X)
        else None
    )
    fold_future_cases: pd.Series | None = None
    fold_forward_positions: pd.Series | None = None
    if (
        fold_local_labeling
        and case_ref is not None
        and district_ref is not None
    ):
        fold_future_cases = _build_future_case_series(
            case_series=case_ref,
            district_series=district_ref,
            temporal_series=temporal_ref,
        )
        fold_forward_positions = _build_forward_horizon_position_series(
            district_series=district_ref,
            temporal_series=temporal_ref,
            horizon_steps=int(getattr(cv_cfg, "label_horizon_steps", 1)),
        )

    if config.enable_temporal_cv:
        folds_started = 0
        folds_skipped_single_class = 0
        folds_with_predictions = 0
        total_yielded_folds = int(len(yielded_fold_ledger))
        for fold_id, (train_idx, valid_idx) in enumerate(generate_time_splits_fn(training_frame, cv_cfg), start=1):
            folds_started += 1
            fold_meta = yielded_fold_ledger[fold_id - 1] if fold_id <= total_yielded_folds else {}
            remaining_folds = max(total_yielded_folds - fold_id, 0)
            LOGGER.info(
                "Baseline CV fold %d/%d START | valid_year=%s, train=%s-%s, rows_train=%d, rows_valid=%d, remaining_after=%d",
                int(fold_id),
                int(total_yielded_folds),
                fold_meta.get("valid_year", "unknown"),
                fold_meta.get("train_start_year", "unknown"),
                fold_meta.get("train_end_year", "unknown"),
                int(len(train_idx)),
                int(len(valid_idx)),
                int(remaining_folds),
            )
            fold_dir = output_root / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_labels = pd.Index(train_idx)
            valid_labels = pd.Index(valid_idx)
            train_positions = X.index.get_indexer(train_labels)
            valid_positions = X.index.get_indexer(valid_labels)
            if (train_positions < 0).any() or (valid_positions < 0).any():
                raise RuntimeError(
                    f"Fold index alignment failed for fold={fold_id}; received labels not present in training frame index"
                )

            purged_train_rows = 0
            if fold_future_cases is not None and fold_forward_positions is not None and train_positions.size and valid_positions.size:
                valid_position_set = set(int(value) for value in valid_positions.tolist())
                forward_for_train = pd.to_numeric(fold_forward_positions.loc[train_labels], errors="coerce")
                purge_mask = forward_for_train.isin(valid_position_set).to_numpy(dtype=bool)
                purged_train_rows = int(purge_mask.sum())
                if purged_train_rows > 0:
                    train_labels = train_labels[~purge_mask]
                    train_positions = train_positions[~purge_mask]
            LOGGER.info(
                "Baseline CV fold %d/%d purge guard | horizon_steps=%d, purged_train_rows=%d, rows_train_after_purge=%d",
                int(fold_id),
                int(total_yielded_folds),
                int(max(1, int(getattr(cv_cfg, "label_horizon_steps", 1)))),
                int(purged_train_rows),
                int(len(train_positions)),
            )

            X_train = X.loc[train_labels]
            X_valid = X.loc[valid_labels]
            y_train = y.loc[train_labels]
            y_valid = y.loc[valid_labels]

            if fold_local_climate_imputation:
                X_train, X_valid = _impute_fold_climate_features(X_train, X_valid)

            if fold_future_cases is not None:
                y_train, y_valid = _derive_fold_targets_from_train_threshold(
                    case_series=case_ref,
                    district_series=district_ref,
                    future_case_series=fold_future_cases,
                    train_positions=train_positions,
                    valid_positions=valid_positions,
                    selected_percentile=int(selected_percentile),
                )

            valid_index = valid_labels

            if pd.to_numeric(y_train, errors="coerce").dropna().nunique() <= 1:
                folds_skipped_single_class += 1
                LOGGER.info(
                    "Skipping fold_%d because training labels are single-class | remaining_after=%d",
                    fold_id,
                    int(remaining_folds),
                )
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
                        "rows_train": int(len(train_positions)),
                        "rows_valid": int(len(valid_idx)),
                        "accuracy": _accuracy(y_valid, y_prob),
                        "brier": float("nan"),
                        "positive_rate": float(pd.to_numeric(y_prob, errors="coerce").fillna(0.0).mean()),
                    }
                )

            if fold_predictions:
                folds_with_predictions += 1
                pd.DataFrame(fold_predictions, index=valid_index).to_csv(
                    fold_dir / "predictions.csv",
                    index=True,
                )
                LOGGER.info(
                    "Baseline CV fold %d/%d END | models_trained=%d, remaining_after=%d",
                    int(fold_id),
                    int(total_yielded_folds),
                    int(len(fold_predictions)),
                    int(remaining_folds),
                )
            else:
                LOGGER.info(
                    "Baseline CV fold %d/%d END | no_models_trained, remaining_after=%d",
                    int(fold_id),
                    int(total_yielded_folds),
                    int(remaining_folds),
                )

        LOGGER.info(
            "Baseline CV summary | yielded=%d, started=%d, folds_with_predictions=%d, skipped_single_class=%d",
            int(total_yielded_folds),
            int(folds_started),
            int(folds_with_predictions),
            int(folds_skipped_single_class),
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
