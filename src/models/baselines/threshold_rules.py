"""Threshold-rule baselines and thresholding helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class DistrictPercentileThresholdRule:
    """EWARS-style district percentile outbreak rule.

    The rule learns district-specific case thresholds on train data and emits
    a binary alert probability in {0, 1} for each row.
    """

    percentile: float = 0.75
    district_column: str = "district"
    case_column: str = "cases"
    district_thresholds_: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    global_threshold_: float = 0.0
    fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "DistrictPercentileThresholdRule":
        """Fit district-level percentile thresholds."""
        if self.case_column not in df.columns:
            raise ValueError(f"Missing case column '{self.case_column}' for threshold rule")
        if self.district_column not in df.columns:
            LOGGER.warning(
                "Missing district column '%s'; using global percentile threshold",
                self.district_column,
            )

        case_values = pd.to_numeric(df[self.case_column], errors="coerce")
        self.global_threshold_ = float(case_values.quantile(self.percentile)) if case_values.notna().any() else 0.0

        if self.district_column in df.columns:
            grouped = (
                pd.DataFrame(
                    {
                        self.district_column: df[self.district_column],
                        self.case_column: case_values,
                    }
                )
                .dropna(subset=[self.case_column])
                .groupby(self.district_column, dropna=False)[self.case_column]
            )
            self.district_thresholds_ = grouped.quantile(self.percentile)
        else:
            self.district_thresholds_ = pd.Series(dtype="float64")

        self.fitted_ = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Predict binary alert probability as a conservative rule score."""
        if not self.fitted_:
            raise RuntimeError("Threshold rule must be fit before prediction")
        if self.case_column not in df.columns:
            LOGGER.warning("Missing case column '%s'; returning zero risk", self.case_column)
            return pd.Series(0.0, index=df.index, name="rule_based_threshold")

        cases = pd.to_numeric(df[self.case_column], errors="coerce").fillna(0.0)
        if self.district_column in df.columns and not self.district_thresholds_.empty:
            thresholds = df[self.district_column].map(self.district_thresholds_).fillna(self.global_threshold_)
        else:
            thresholds = pd.Series(self.global_threshold_, index=df.index)
        proba = (cases >= thresholds).astype(float)
        proba.name = "rule_based_threshold"
        return proba


def apply_threshold(probabilities: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Convert probabilities into binary alerts."""
    return (pd.to_numeric(probabilities, errors="coerce").fillna(0.0) >= threshold).astype(int)
