"""Model registry helpers for baseline artifacts and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Dict

from src.models.baselines.baseline_models import BaselineModel, MODEL_ALIASES, get_baseline_model

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ORDER: tuple[str, ...] = (
    "rule_based_threshold",
    "logistic_regression",
    "poisson_regression",
    "negative_binomial_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
)


@dataclass(frozen=True)
class BaselineModelSpec:
    """Registry metadata for one baseline model family."""

    name: str
    description: str
    optional_dependency: str | None = None


MODEL_SPECS: dict[str, BaselineModelSpec] = {
    "rule_based_threshold": BaselineModelSpec(
        name="rule_based_threshold",
        description="District percentile threshold rule baseline",
    ),
    "logistic_regression": BaselineModelSpec(
        name="logistic_regression",
        description="L2-regularized logistic classifier",
    ),
    "poisson_regression": BaselineModelSpec(
        name="poisson_regression",
        description="Poisson GLM-style baseline",
    ),
    "negative_binomial_regression": BaselineModelSpec(
        name="negative_binomial_regression",
        description="Negative binomial alternative for overdispersion",
        optional_dependency="statsmodels",
    ),
    "random_forest": BaselineModelSpec(
        name="random_forest",
        description="Random forest classifier baseline",
    ),
    "xgboost": BaselineModelSpec(
        name="xgboost",
        description="Gradient boosting baseline with XGBoost",
        optional_dependency="xgboost",
    ),
    "lightgbm": BaselineModelSpec(
        name="lightgbm",
        description="Gradient boosting baseline with LightGBM",
        optional_dependency="lightgbm",
    ),
}


def normalize_model_name(name: str) -> str:
    """Normalize user-provided model name/alias into canonical form."""
    return MODEL_ALIASES.get(name.lower(), name.lower())


def list_default_model_names() -> list[str]:
    """Return canonical baseline model list in default benchmark order."""
    return list(DEFAULT_MODEL_ORDER)


def validate_model_names(model_names: list[str]) -> list[str]:
    """Validate and normalize model names, dropping unknown names with logging."""
    normalized: list[str] = []
    for raw_name in model_names:
        model_name = normalize_model_name(raw_name)
        if model_name not in MODEL_SPECS:
            LOGGER.warning("Unknown model '%s' skipped", raw_name)
            continue
        normalized.append(model_name)
    return normalized


@dataclass
class ModelRegistry:
    """In-memory registry for trained baseline model instances."""

    models: Dict[str, BaselineModel] = field(default_factory=dict)

    def register(self, name: str, model: BaselineModel) -> None:
        """Register (or replace) a trained model by name."""
        canonical_name = normalize_model_name(name)
        self.models[canonical_name] = model

    def get(self, name: str) -> BaselineModel:
        """Retrieve a trained model by name."""
        canonical_name = normalize_model_name(name)
        return self.models[canonical_name]

    def ensure(self, name: str, *, random_state: int = 42) -> BaselineModel:
        """Return existing model or create a fresh model instance."""
        canonical_name = normalize_model_name(name)
        if canonical_name not in self.models:
            self.models[canonical_name] = get_baseline_model(canonical_name, random_state=random_state)
        return self.models[canonical_name]
