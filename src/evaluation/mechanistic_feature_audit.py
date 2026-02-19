"""Mechanistic feature usefulness audit for outbreak prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pointbiserialr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from config.paths import ensure_directories

LOGGER = logging.getLogger(__name__)

DEFAULT_MECHANISTIC_FEATURES: tuple[str, ...] = (
    "temp_anomaly",
    "degree_days_20",
    "rainfall_4wk",
    "lai_anomaly",
    "temp_optimal",
    "temp_celsius",
    "temp_rain_interaction",
)


@dataclass(frozen=True)
class AuditArtifacts:
    scores_csv: Path
    report_md: Path
    figure_png: Path


def _safe_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Required column '{column}' not found")
    return df[column]


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _minmax(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    minimum = numeric.min(skipna=True)
    maximum = numeric.max(skipna=True)
    if pd.isna(minimum) or pd.isna(maximum) or maximum <= minimum:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (numeric - minimum) / (maximum - minimum)


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _prepare_time_index(df: pd.DataFrame, date_col: str | None, year_col: str | None, month_col: str | None) -> pd.Series:
    if date_col is not None:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed

    if year_col is None:
        raise ValueError("Could not infer temporal ordering; no usable date/year column found")

    year_values = pd.to_numeric(df[year_col], errors="coerce")
    if month_col is not None:
        month_values = pd.to_numeric(df[month_col], errors="coerce").fillna(1)
    else:
        month_values = pd.Series(np.ones(len(df), dtype=float), index=df.index)

    timestamp = pd.to_datetime(
        {
            "year": year_values.fillna(1970).astype(int),
            "month": month_values.clip(lower=1, upper=12).astype(int),
            "day": 1,
        },
        errors="coerce",
    )
    return timestamp


def _feature_quality_diagnostics(df: pd.DataFrame, feature: str, district_col: str | None, time_col: str | None) -> dict[str, float]:
    values = _to_numeric(df[feature])
    missing_rate = float(values.isna().mean())
    non_missing = values.dropna()
    variance = float(non_missing.var(ddof=0)) if len(non_missing) > 1 else 0.0
    unique_count = int(non_missing.nunique())
    uniqueness_rate = float(unique_count / max(len(non_missing), 1))

    district_cv = np.nan
    temporal_cv = np.nan

    if district_col and district_col in df.columns and len(non_missing) > 0:
        district_means = df.assign(_value=values).groupby(district_col, dropna=False)["_value"].mean()
        dm = district_means.dropna()
        if len(dm) > 1 and float(dm.mean()) != 0.0:
            district_cv = float(dm.std(ddof=0) / abs(dm.mean()))

    if time_col and time_col in df.columns and len(non_missing) > 0:
        temporal_means = df.assign(_value=values).groupby(time_col, dropna=False)["_value"].mean()
        tm = temporal_means.dropna()
        if len(tm) > 1 and float(tm.mean()) != 0.0:
            temporal_cv = float(tm.std(ddof=0) / abs(tm.mean()))

    cv_values = [value for value in (district_cv, temporal_cv) if np.isfinite(value)]
    if variance <= 1e-12 or uniqueness_rate <= 0.01 or not cv_values:
        stability_score = 0.0
    else:
        stability_raw = float(np.mean(cv_values))
        stability_score = float(1.0 / (1.0 + stability_raw))

    return {
        "missing_rate": missing_rate,
        "variance": variance,
        "uniqueness_rate": uniqueness_rate,
        "district_cv": float(district_cv) if np.isfinite(district_cv) else np.nan,
        "temporal_cv": float(temporal_cv) if np.isfinite(temporal_cv) else np.nan,
        "stability_score": stability_score,
    }


def _univariate_signals(df: pd.DataFrame, feature: str, target_col: str) -> dict[str, float]:
    x = _to_numeric(df[feature])
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 12:
        return {
            "spearman": np.nan,
            "point_biserial": np.nan,
            "fisher_score": np.nan,
            "mutual_info": np.nan,
        }

    xv = x.loc[valid]
    yv = y.loc[valid].astype(int)
    if xv.nunique(dropna=True) < 2:
        return {
            "spearman": 0.0,
            "point_biserial": 0.0,
            "fisher_score": 0.0,
            "mutual_info": 0.0,
        }
    if yv.nunique() < 2:
        return {
            "spearman": np.nan,
            "point_biserial": np.nan,
            "fisher_score": np.nan,
            "mutual_info": np.nan,
        }

    spearman_value = float(spearmanr(xv, yv, nan_policy="omit").correlation)
    point_biserial_value = float(pointbiserialr(yv, xv).statistic)

    group0 = xv[yv == 0]
    group1 = xv[yv == 1]
    pooled_var = float(xv.var(ddof=0))
    if pooled_var <= 0 or len(group0) == 0 or len(group1) == 0:
        fisher_score = 0.0
    else:
        fisher_score = float(((group1.mean() - group0.mean()) ** 2) / pooled_var)

    try:
        mi_value = float(mutual_info_classif(xv.to_frame(name=feature), yv, discrete_features=False, random_state=42)[0])
    except Exception:
        mi_value = np.nan

    return {
        "spearman": spearman_value,
        "point_biserial": point_biserial_value,
        "fisher_score": fisher_score,
        "mutual_info": mi_value,
    }


def _tree_importance(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    time_index: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    model_df = df[features + [target_col]].copy()
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df = model_df.assign(_time=time_index)
    model_df = model_df.dropna(subset=[target_col, "_time"]).sort_values("_time")

    if model_df.empty:
        raise ValueError("No rows available for model-based importance after filtering.")

    split_idx = int(len(model_df) * 0.8)
    split_idx = min(max(split_idx, 1), len(model_df) - 1)
    train_df = model_df.iloc[:split_idx].copy()
    valid_df = model_df.iloc[split_idx:].copy()

    train_x = train_df[features].apply(pd.to_numeric, errors="coerce")
    valid_x = valid_df[features].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(numeric_only=True)
    train_x = train_x.fillna(medians)
    valid_x = valid_x.fillna(medians)

    train_y = train_df[target_col].astype(int)
    valid_y = valid_df[target_col].astype(int)

    if train_y.nunique() < 2 or valid_y.nunique() < 2:
        raise ValueError("Time split has a single-class train or validation target; cannot estimate robust importance.")

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    clf.fit(train_x, train_y)

    importances = pd.Series(clf.feature_importances_, index=features, dtype=float)

    try:
        perm = permutation_importance(
            clf,
            valid_x,
            valid_y,
            n_repeats=20,
            random_state=42,
            n_jobs=-1,
            scoring="average_precision",
        )
        permutation_values = pd.Series(perm.importances_mean, index=features, dtype=float)
    except Exception:
        permutation_values = pd.Series(np.nan, index=features, dtype=float)

    proba = clf.predict_proba(valid_x)[:, 1]
    diagnostics = {
        "train_rows": float(len(train_df)),
        "valid_rows": float(len(valid_df)),
        "valid_positive_rate": float(valid_y.mean()),
        "valid_pr_auc": float(average_precision_score(valid_y, proba)),
        "valid_roc_auc": float(roc_auc_score(valid_y, proba)),
    }

    return importances, permutation_values, diagnostics


def _pick_mechanistic_features(df: pd.DataFrame) -> list[str]:
    features = [feature for feature in DEFAULT_MECHANISTIC_FEATURES if feature in df.columns]
    if not features:
        raise ValueError("No mechanistic features found in feature matrix")
    return features


def _render_importance_plot(score_df: pd.DataFrame, output_path: Path) -> None:
    ordered = score_df.sort_values("combined_usefulness_score", ascending=False)
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=ordered, x="combined_usefulness_score", y="feature", color="#4C72B0")
    plt.title("Mechanistic Feature Usefulness Score")
    plt.xlabel("Combined usefulness score")
    plt.ylabel("Feature")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def _build_markdown_report(
    score_df: pd.DataFrame,
    report_path: Path,
    label_agreement: float,
    label_overlap_rows: int,
    model_diagnostics: dict[str, float],
) -> None:
    top = score_df.sort_values("combined_usefulness_score", ascending=False).head(5)
    bottom = score_df.sort_values("combined_usefulness_score", ascending=True).head(5)

    weak = score_df[
        (score_df["missing_rate"] > 0.20)
        | (score_df["uniqueness_rate"] < 0.05)
        | (score_df["stability_score"] < 0.40)
        | (score_df["combined_usefulness_score"] < score_df["combined_usefulness_score"].median())
    ].copy()

    caveats: list[str] = []
    if label_overlap_rows == 0:
        caveats.append("No overlap rows found for direct label consistency check between feature matrix and labeled dataset.")
    else:
        caveats.append(
            f"Label agreement between feature matrix and labeled dataset on overlap rows: {label_agreement:.2%} ({label_overlap_rows} rows)."
        )
    if float(model_diagnostics.get("valid_positive_rate", np.nan)) < 0.05:
        caveats.append("Validation period has low outbreak prevalence (<5%), which can increase ranking uncertainty.")
    if weak["missing_rate"].max(skipna=True) > 0.30:
        caveats.append("At least one mechanistic feature has >30% missingness, reducing reliability.")

    lines: list[str] = []
    lines.append("# Mechanistic Feature Audit")
    lines.append("")
    lines.append("## Summary")
    lines.append(
        "This audit evaluates mechanistic feature quality and predictive usefulness for `outbreak_label` using univariate tests and a time-split tree model."
    )
    lines.append("")
    lines.append("## Top Useful Mechanistic Features")
    for _, row in top.iterrows():
        lines.append(
            "- "
            f"**{row['feature']}**: score={row['combined_usefulness_score']:.3f}, "
            f"tree={row['tree_importance']:.3f}, perm={row['permutation_importance']:.3f}, "
            f"MI={row['mutual_info']:.3f}, missing={row['missing_rate']:.1%}"
        )
    lines.append("")
    lines.append("## Bottom / Unstable Features")
    for _, row in bottom.iterrows():
        lines.append(
            "- "
            f"**{row['feature']}**: score={row['combined_usefulness_score']:.3f}, "
            f"stability={row['stability_score']:.3f}, uniqueness={row['uniqueness_rate']:.3f}, "
            f"missing={row['missing_rate']:.1%}"
        )
    lines.append("")
    lines.append("## Weakness Flags")
    if weak.empty:
        lines.append("- No mechanistic features crossed the weak/unreliable thresholds.")
    else:
        for _, row in weak.sort_values("combined_usefulness_score").iterrows():
            reasons: list[str] = []
            if row["missing_rate"] > 0.20:
                reasons.append("high missingness")
            if row["uniqueness_rate"] < 0.05:
                reasons.append("low uniqueness")
            if row["stability_score"] < 0.40:
                reasons.append("temporal/district instability")
            if row["combined_usefulness_score"] < score_df["combined_usefulness_score"].median():
                reasons.append("below-median combined usefulness")
            lines.append(f"- **{row['feature']}**: {', '.join(reasons)}")
    lines.append("")
    lines.append("## Model Split Diagnostics")
    lines.append(f"- Train rows: {int(model_diagnostics.get('train_rows', 0))}")
    lines.append(f"- Validation rows: {int(model_diagnostics.get('valid_rows', 0))}")
    lines.append(f"- Validation outbreak rate: {model_diagnostics.get('valid_positive_rate', np.nan):.2%}")
    lines.append(f"- Validation PR-AUC (mechanistic-only model): {model_diagnostics.get('valid_pr_auc', np.nan):.3f}")
    lines.append(f"- Validation ROC-AUC (mechanistic-only model): {model_diagnostics.get('valid_roc_auc', np.nan):.3f}")
    lines.append("")
    lines.append("## Data Quality Caveats")
    for caveat in caveats:
        lines.append(f"- {caveat}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    feature_matrix_path: Path | None = None,
    labeled_path: Path | None = None,
) -> AuditArtifacts:
    paths = ensure_directories()
    feature_path = feature_matrix_path or (paths.data_features / "feature_matrix.csv")
    labels_path = labeled_path or (paths.data_processed / "epiclim_labeled.csv")

    LOGGER.info("Loading feature matrix: %s", feature_path)
    feature_df = pd.read_csv(feature_path)
    LOGGER.info("Loading labeled dataset: %s", labels_path)
    labeled_df = pd.read_csv(labels_path)

    mechanistic_features = _pick_mechanistic_features(feature_df)
    LOGGER.info("Mechanistic features found: %s", ", ".join(mechanistic_features))

    target_col = "outbreak_label"
    _safe_series(feature_df, target_col)

    district_col = _first_existing(feature_df, ("district",))
    date_col = _first_existing(feature_df, ("date", "week_of_outbreak"))
    month_col = _first_existing(feature_df, ("month", "mon"))
    year_col = _first_existing(feature_df, ("year",))
    time_index = _prepare_time_index(feature_df, date_col=date_col, year_col=year_col, month_col=month_col)

    if date_col is not None:
        time_group_col = "_audit_time_group"
        feature_df[time_group_col] = pd.to_datetime(feature_df[date_col], errors="coerce").dt.to_period("M").astype(str)
    else:
        time_group_col = month_col if month_col is not None else year_col

    rows: list[dict[str, float | str]] = []
    for feature in mechanistic_features:
        quality = _feature_quality_diagnostics(
            feature_df,
            feature=feature,
            district_col=district_col,
            time_col=time_group_col,
        )
        univariate = _univariate_signals(feature_df, feature=feature, target_col=target_col)
        row: dict[str, float | str] = {"feature": feature}
        row.update(quality)
        row.update(univariate)
        rows.append(row)

    score_df = pd.DataFrame(rows)

    tree_importance, perm_importance, model_diagnostics = _tree_importance(
        feature_df,
        features=mechanistic_features,
        target_col=target_col,
        time_index=time_index,
    )

    score_df["tree_importance"] = score_df["feature"].map(tree_importance)
    score_df["permutation_importance"] = score_df["feature"].map(perm_importance)

    score_df["quality_component"] = (
        0.35 * (1.0 - score_df["missing_rate"].fillna(1.0))
        + 0.20 * _minmax(score_df["uniqueness_rate"].fillna(0.0))
        + 0.20 * score_df["stability_score"].fillna(0.0)
        + 0.25 * _minmax(np.log1p(score_df["variance"].fillna(0.0)))
    )
    score_df["association_component"] = (
        0.20 * _minmax(score_df["spearman"].abs().fillna(0.0))
        + 0.20 * _minmax(score_df["point_biserial"].abs().fillna(0.0))
        + 0.20 * _minmax(score_df["fisher_score"].fillna(0.0))
        + 0.20 * _minmax(score_df["mutual_info"].fillna(0.0))
        + 0.10 * _minmax(score_df["tree_importance"].fillna(0.0))
        + 0.10 * _minmax(score_df["permutation_importance"].fillna(0.0))
    )
    score_df["combined_usefulness_score"] = 0.35 * score_df["quality_component"] + 0.65 * score_df["association_component"]

    score_df = score_df.sort_values("combined_usefulness_score", ascending=False).reset_index(drop=True)

    overlap_keys = [column for column in ("district", "date") if column in feature_df.columns and column in labeled_df.columns]
    label_agreement = np.nan
    overlap_rows = 0
    if overlap_keys:
        left = feature_df[overlap_keys + [target_col]].copy()
        right = labeled_df[overlap_keys + [target_col]].copy()
        merged = left.merge(right, on=overlap_keys, how="inner", suffixes=("_feature", "_labeled"))
        overlap_rows = len(merged)
        if overlap_rows > 0:
            both = merged[[f"{target_col}_feature", f"{target_col}_labeled"]].dropna()
            if len(both) > 0:
                label_agreement = float((both[f"{target_col}_feature"].astype(int) == both[f"{target_col}_labeled"].astype(int)).mean())

    scores_path = paths.outputs_metrics / "mechanistic_feature_scores.csv"
    report_path = paths.outputs_reports / "mechanistic_feature_audit.md"
    figure_path = paths.outputs_figures / "mechanistic_feature_importance.png"

    score_df.to_csv(scores_path, index=False)
    _render_importance_plot(score_df, output_path=figure_path)
    _build_markdown_report(
        score_df=score_df,
        report_path=report_path,
        label_agreement=label_agreement,
        label_overlap_rows=overlap_rows,
        model_diagnostics=model_diagnostics,
    )

    LOGGER.info("Saved mechanistic audit score table to %s", scores_path)
    LOGGER.info("Saved mechanistic audit figure to %s", figure_path)
    LOGGER.info("Saved mechanistic audit report to %s", report_path)

    return AuditArtifacts(scores_csv=scores_path, report_md=report_path, figure_png=figure_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    artifacts = run_audit()
    print("mechanistic_feature_scores:", artifacts.scores_csv)
    print("mechanistic_feature_audit_report:", artifacts.report_md)
    print("mechanistic_feature_importance_figure:", artifacts.figure_png)


if __name__ == "__main__":
    main()
