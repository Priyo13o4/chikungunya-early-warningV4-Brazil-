from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import importlib
import inspect
import logging
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np

from src.models.baselines.cv_splitter import TimeSeriesCVConfig
from src.models.baselines.model_registry import list_default_model_names
from src.pipeline_runtime.compute_backend import ComputeBackendConfig
from src.pipeline_runtime.compute_backend import parse_compute_backend_config as parse_compute_backend_config_runtime

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BayesianSubsetConfig:
    enabled: bool = False
    max_rows: int = 0
    strategy: str = "recent_years"


@dataclass(frozen=True)
class MemoryOptimizationConfig:
    mode: str = "off"
    train_year_window: tuple[int, int] | None = None
    district_shard_count: int | None = None
    district_shard_index: int | None = None
    bayesian_subset: BayesianSubsetConfig = field(default_factory=BayesianSubsetConfig)

_ADAPTER_CALLABLE_KEYS: tuple[str, ...] = (
    "load_data",
    "label_outbreaks",
    "build_feature_matrix",
    "build_fold_ledger",
    "generate_time_splits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chikungunya early warning pipeline")
    default_baseline_models = list_default_model_names()
    parser.add_argument("--model-config", type=Path, default=Path("config/model_config.yaml"), help="Path to model config YAML")
    parser.add_argument(
        "--adapter-config",
        type=Path,
        default=None,
        help="Optional adapter config YAML with dynamic import paths for load/label/features/CV callables",
    )
    parser.add_argument("--cv-config", type=Path, default=Path("config/cv_config.yaml"), help="Path to temporal CV config YAML")
    parser.add_argument("--raw-data", type=Path, default=Path("data/raw/Epiclim_Final_data.csv"), help="Path to raw CSV data file")
    parser.add_argument(
        "--population-data",
        type=Path,
        default=None,
        help="Optional population/census path (CSV/XLS/XLSX) for merge stage; auto-discovered from data/raw when omitted",
    )
    parser.add_argument("--start-year", type=int, default=2009, help="Start year for cleaning/filtering")
    parser.add_argument("--end-year", type=int, default=2019, help="End year for cleaning/filtering")
    parser.add_argument("--selected-percentile", type=int, default=75, help="District percentile used for canonical outbreak_label")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline model training/prediction")
    parser.add_argument("--skip-bayesian", action="store_true", help="Skip Bayesian model track")
    parser.add_argument("--skip-visualizations", action="store_true", help="Skip visualization artifact generation")
    parser.add_argument("--decision-cost", type=float, default=0.2, help="Cost parameter for cost-loss decision action")
    parser.add_argument("--decision-loss", type=float, default=1.0, help="Loss parameter for cost-loss decision action")
    parser.add_argument(
        "--lead-time-max-lookback-steps",
        type=int,
        default=8,
        help="Maximum prior steps credited for lead-time scoring (alerts older than horizon get zero credit).",
    )
    parser.add_argument(
        "--strict-bayesian-deps",
        action="store_true",
        help="Fail instead of gracefully skipping when Bayesian dependencies are unavailable",
    )
    parser.add_argument(
        "--strict-feature-gate",
        action="store_true",
        help="Fail pipeline when mechanistic feature quality gate flags near-degenerate features.",
    )
    parser.add_argument("--bayesian-draws", type=int, default=None, help="Override Bayesian posterior draws")
    parser.add_argument("--bayesian-tune", type=int, default=None, help="Override Bayesian warmup/tune steps")
    parser.add_argument("--bayesian-chains", type=int, default=None, help="Override Bayesian chain count")
    parser.add_argument("--bayesian-target-accept", type=float, default=None, help="Override Bayesian NUTS target_accept")
    parser.add_argument("--bayesian-max-treedepth", type=int, default=None, help="Override Bayesian NUTS max_treedepth")
    parser.add_argument(
        "--bayesian-progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show Bayesian sampling progress bar (default: true)",
    )
    parser.add_argument(
        "--bayesian-simplified-mode",
        action="store_true",
        help="Run Bayesian track in simplified mode (removes latent AR(1) state for stability)",
    )
    parser.add_argument(
        "--force-full-bayesian",
        action="store_true",
        help="Force full latent AR Bayesian mode only (disables simplified fallback/retry).",
    )
    parser.add_argument(
        "--export-detailed-csv",
        action="store_true",
        help="Export detailed prediction CSVs under outputs/models (disabled by default)",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=None,
        help=(
            "Baseline model names/aliases to train. "
            "If omitted, uses config/model_config.yaml baseline_models; "
            f"default suite: {', '.join(default_baseline_models)}"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument("--seed", type=int, default=None, help="Global random seed override for baseline and Bayesian tracks")
    return parser.parse_args()


def build_bayesian_config(strict_dependencies: bool, bayesian_settings: dict[str, Any]) -> Any:
    from src.models.bayesian.hierarchical_model import BayesianModelConfig

    return BayesianModelConfig(
        strict_dependencies=strict_dependencies,
        draws=int(bayesian_settings.get("draws", BayesianModelConfig.draws)),
        tune=int(bayesian_settings.get("tune", BayesianModelConfig.tune)),
        chains=int(bayesian_settings.get("chains", BayesianModelConfig.chains)),
        bayesian_progress=bool(bayesian_settings.get("bayesian_progress", BayesianModelConfig.bayesian_progress)),
        target_accept=float(bayesian_settings.get("target_accept", BayesianModelConfig.target_accept)),
        max_treedepth=int(bayesian_settings.get("max_treedepth", BayesianModelConfig.max_treedepth)),
        bayesian_simplified_mode=bool(
            bayesian_settings.get("bayesian_simplified_mode", BayesianModelConfig.bayesian_simplified_mode)
        ),
        max_convergence_retries=int(
            bayesian_settings.get("max_convergence_retries", BayesianModelConfig.max_convergence_retries)
        ),
        divergence_warn_threshold=int(
            bayesian_settings.get("divergence_warn_threshold", BayesianModelConfig.divergence_warn_threshold)
        ),
        rhat_warn_threshold=float(bayesian_settings.get("rhat_warn_threshold", BayesianModelConfig.rhat_warn_threshold)),
        ess_warn_threshold=float(bayesian_settings.get("ess_warn_threshold", BayesianModelConfig.ess_warn_threshold)),
        random_seed=int(bayesian_settings.get("random_seed", BayesianModelConfig.random_seed)),
        force_full_bayesian=bool(bayesian_settings.get("force_full_bayesian", BayesianModelConfig.force_full_bayesian)),
        outbreak_threshold_default_cases=float(
            bayesian_settings.get(
                "outbreak_threshold_default_cases",
                BayesianModelConfig.outbreak_threshold_default_cases,
            )
        ),
        posterior_sample_cap=int(bayesian_settings.get("posterior_sample_cap", BayesianModelConfig.posterior_sample_cap)),
    )


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        LOGGER.warning("Model config not found at %s; using defaults", config_path)
        return {}
    try:
        import yaml
    except Exception as yaml_error:
        LOGGER.warning("PyYAML not available (%s); unable to read %s", yaml_error, config_path)
        return {}

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as load_error:
        LOGGER.warning("Unable to load YAML config from %s: %s", config_path, load_error)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge_dict(existing, value)
        else:
            merged[key] = value
    return merged


def import_callable(path_spec: str) -> Callable[..., Any]:
    spec = str(path_spec).strip()
    if ":" in spec:
        module_name, attribute_name = spec.split(":", 1)
    else:
        module_name, _, attribute_name = spec.rpartition(".")
    if not module_name or not attribute_name:
        raise ValueError(f"Invalid callable import path: '{path_spec}'")
    module = importlib.import_module(module_name)
    loaded = getattr(module, attribute_name, None)
    if not callable(loaded):
        raise TypeError(f"Imported object is not callable: '{path_spec}'")
    return loaded


def resolve_adapter_callable(
    adapter_config: dict[str, Any],
    *,
    key: str,
    default: Callable[..., Any],
) -> Callable[..., Any]:
    configured = adapter_config.get(key)
    if not configured:
        return default
    try:
        loaded = import_callable(str(configured))
    except Exception as import_error:
        LOGGER.warning(
            "Unable to load adapter callable '%s' from key '%s' (%s); using default %s",
            configured,
            key,
            import_error,
            default.__name__,
        )
        return default
    LOGGER.info("Adapter callable enabled for '%s': %s", key, configured)
    return loaded


def is_brazil_adapter_config_active(adapter_config: dict[str, Any]) -> bool:
    if not isinstance(adapter_config, dict) or not adapter_config:
        return False

    callable_specs: list[str] = []
    for key in _ADAPTER_CALLABLE_KEYS:
        value = adapter_config.get(key)
        if value is None:
            continue
        callable_specs.append(str(value).strip())

    if not callable_specs:
        return False
    return all(spec.startswith("projects.brazil_chik.") for spec in callable_specs)


def resolve_cv_config(
    *,
    cv_config_path: Path,
    date_column: str,
    target_column: str,
    end_year: int,
) -> tuple[TimeSeriesCVConfig, dict[str, Any]]:
    raw_cv_config = load_yaml_config(cv_config_path)

    first_valid_year = int(raw_cv_config.get("first_valid_year", 0) or 0)
    last_valid_year = int(raw_cv_config.get("last_valid_year", 0) or 0)
    n_splits = int(raw_cv_config.get("n_splits", 0) or 0)

    if first_valid_year <= 0:
        first_valid_year = max(2009, (end_year - max(1, n_splits) + 1) if n_splits > 0 else 2014)
    if last_valid_year <= 0:
        last_valid_year = end_year

    cv_cfg = TimeSeriesCVConfig(
        date_column=str(raw_cv_config.get("date_column", date_column)),
        target_column=str(raw_cv_config.get("target_column", target_column)),
        start_train_year=int(raw_cv_config.get("start_train_year", 2009)),
        first_valid_year=int(first_valid_year),
        last_valid_year=int(last_valid_year),
        train_window_years=int(raw_cv_config.get("train_window_years", 5)),
        thesis_strict=bool(raw_cv_config.get("thesis_strict", False)),
        skip_single_class_folds=bool(raw_cv_config.get("skip_single_class_folds", True)),
        minimum_evaluated_folds=int(raw_cv_config.get("minimum_evaluated_folds", max(1, min(3, n_splits or 5)))),
    )
    effective = {
        **raw_cv_config,
        **asdict(cv_cfg),
    }
    return cv_cfg, effective


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def parse_compute_backend_config(raw_model_config: dict[str, Any]) -> ComputeBackendConfig:
    return parse_compute_backend_config_runtime(raw_model_config)


def parse_memory_optimization_config(raw_model_config: dict[str, Any] | None) -> MemoryOptimizationConfig:
    if not isinstance(raw_model_config, dict):
        return MemoryOptimizationConfig()

    raw_memory = raw_model_config.get("memory_optimization")
    if not isinstance(raw_memory, dict):
        return MemoryOptimizationConfig()

    mode = str(raw_memory.get("mode", "off")).strip().lower()
    if mode not in {"off", "year_window", "district_shard", "hybrid"}:
        LOGGER.warning("Invalid memory_optimization.mode '%s'; defaulting to 'off'", mode)
        mode = "off"

    train_year_window: tuple[int, int] | None = None
    raw_window = raw_memory.get("train_year_window")
    if isinstance(raw_window, (list, tuple)) and len(raw_window) == 2:
        try:
            start_year = int(raw_window[0])
            end_year = int(raw_window[1])
            if start_year > end_year:
                start_year, end_year = end_year, start_year
            train_year_window = (start_year, end_year)
        except Exception:
            LOGGER.warning("Invalid memory_optimization.train_year_window '%s'; ignoring", raw_window)

    district_shard_count: int | None = None
    raw_count = raw_memory.get("district_shard_count")
    if raw_count is not None:
        try:
            district_shard_count = int(raw_count)
        except Exception:
            LOGGER.warning("Invalid memory_optimization.district_shard_count '%s'; ignoring", raw_count)
            district_shard_count = None
    if district_shard_count is not None and district_shard_count <= 0:
        LOGGER.warning("Non-positive district_shard_count=%s; ignoring", district_shard_count)
        district_shard_count = None

    district_shard_index: int | None = None
    raw_index = raw_memory.get("district_shard_index")
    if raw_index is not None:
        try:
            district_shard_index = int(raw_index)
        except Exception:
            LOGGER.warning("Invalid memory_optimization.district_shard_index '%s'; ignoring", raw_index)
            district_shard_index = None

    if district_shard_count is not None and district_shard_index is None:
        district_shard_index = 0
    if district_shard_count is not None and district_shard_index is not None:
        district_shard_index = district_shard_index % district_shard_count

    raw_subset = raw_memory.get("bayesian_subset")
    if not isinstance(raw_subset, dict):
        raw_subset = {}
    subset_strategy = str(raw_subset.get("strategy", "recent_years")).strip().lower()
    if subset_strategy not in {"recent_years", "top_cases", "random_stratified"}:
        LOGGER.warning(
            "Invalid memory_optimization.bayesian_subset.strategy '%s'; defaulting to 'recent_years'",
            subset_strategy,
        )
        subset_strategy = "recent_years"
    subset_max_rows = int(raw_subset.get("max_rows", 0) or 0)
    if subset_max_rows < 0:
        subset_max_rows = 0
    bayesian_subset = BayesianSubsetConfig(
        enabled=bool(raw_subset.get("enabled", False)),
        max_rows=int(subset_max_rows),
        strategy=subset_strategy,
    )

    return MemoryOptimizationConfig(
        mode=mode,
        train_year_window=train_year_window,
        district_shard_count=district_shard_count,
        district_shard_index=district_shard_index,
        bayesian_subset=bayesian_subset,
    )


def call_load_data_compat(
    load_data_callable: Callable[..., Any],
    raw_data_path: Path,
    population_data_path: Path | None,
    *,
    start_year: int,
    end_year: int,
    discovery_dir: Path,
) -> tuple[Any, Any]:
    signature = inspect.signature(load_data_callable)
    parameters = signature.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    kwargs: dict[str, Any] = {"discovery_dir": discovery_dir}
    if "start_year" in parameters or accepts_var_kwargs:
        kwargs["start_year"] = int(start_year)
    if "end_year" in parameters or accepts_var_kwargs:
        kwargs["end_year"] = int(end_year)

    try:
        return load_data_callable(raw_data_path, population_data_path, **kwargs)
    except TypeError as call_error:
        fallback_kwargs = {"discovery_dir": discovery_dir}
        unsupported_kw_error = "unexpected keyword" in str(call_error).lower() and (
            "start_year" in str(call_error) or "end_year" in str(call_error)
        )
        if kwargs != fallback_kwargs and unsupported_kw_error:
            LOGGER.info(
                "Loader callable rejected year bounds (%s); falling back to legacy loader call shape",
                call_error,
            )
            return load_data_callable(raw_data_path, population_data_path, **fallback_kwargs)
        raise


def call_label_outbreaks_compat(
    label_callable: Callable[..., Any],
    df: Any,
    *,
    selected_percentile: int,
    use_percentile_labels: bool,
    cv_config: TimeSeriesCVConfig,
    strict_mode: bool,
) -> Any:
    signature = inspect.signature(label_callable)
    parameters = signature.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    kwargs: dict[str, Any] = {
        "selected_percentile": selected_percentile,
        "use_percentile_labels": use_percentile_labels,
        "date_column": cv_config.date_column,
        "first_valid_year": int(cv_config.first_valid_year),
        "last_valid_year": int(cv_config.last_valid_year),
        "start_train_year": int(cv_config.start_train_year),
        "train_window_years": int(cv_config.train_window_years),
        "threshold_scope": "train_fold",
    }
    if not accepts_var_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}

    try:
        return label_callable(df, **kwargs)
    except TypeError as call_error:
        unsupported_kw_error = "unexpected keyword" in str(call_error).lower()
        if kwargs and unsupported_kw_error:
            if strict_mode:
                raise RuntimeError(
                    "Label callable does not support train-fold leakage-control args under strict mode "
                    f"(error: {call_error})"
                ) from call_error
            LOGGER.info(
                "Label callable rejected advanced args (%s); falling back to legacy label call shape",
                call_error,
            )
            return label_callable(
                df,
                selected_percentile=selected_percentile,
                use_percentile_labels=use_percentile_labels,
            )
        raise
