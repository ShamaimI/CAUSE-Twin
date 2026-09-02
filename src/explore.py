"""
CAUSE-Twin: Exploratory Data Analysis Module (Module 3)
Purpose: Generate empirical evidence for the causal argument (confounding between
resource conditions), sanity-check panel trends over time, check whether missingness
is correlated with development level, detect robust statistical outliers via
Median Absolute Deviation (MAD), and evaluate distribution shape.

Design Principles & Methodological Protections:
  1. Robust Outlier Metric: Uses Modified Z-scores based on Median Absolute
     Deviation (MAD) rather than standard deviation to prevent heavy-tail shock inflation.
  2. Genuine-Only Outlier Detection: Outlier flags for forward-filled indicators
     are evaluated solely on genuine survey observations (_is_filled == False).
  3. Post-Hoc Qualitative Annotation: KNOWN_DATA_QUALITY_NOTES provides human context
     for reporting. It strictly NEVER alters data points, drops rows, or modifies
     downstream model matrices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"

# Post-hoc qualitative annotation layer ONLY — documented for transparency.
# Strictly NEVER modifies data points, inclusion criteria, or downstream model matrices.
KNOWN_DATA_QUALITY_NOTES = {
    ("OMN", 2008): "Plausible Methodological Artifact (Unverified against internal logs): "
                    "literacy_rate jumped +20.5 points in one year — likely a new census/survey "
                    "round replacing an older estimate discontinuously.",
    ("EGY", 2021): "Plausible Methodological Artifact (Unverified against internal logs): "
                    "literacy_rate jumped +10.9 points in one year — likely a survey methodology discontinuity.",
    ("SYR", 2012): "Confirmed Major Shock: mortality_u5 spike (+12.6) coincides with the "
                    "onset of the Syrian Civil War — real shock, not a data entry error.",
    ("COD", 2010): "Plausible Methodological Artifact: Observed U5MR shift aligns temporally "
                    "with DHS/MICS survey harmonization windows.",
    ("LSO", 2004): "Plausible Methodological Artifact: Discontinuity corresponds to post-census "
                    "mortality weighting revisions.",
}

INDICATORS = [
    "mortality_u5",
    "nutrition_stunting",
    "sanitation_basic",
    "gdp_per_capita",
    "literacy_rate",
]

DEFAULT_MAD_THRESHOLD = 3.5  # Modified Z-score threshold using MAD


def load_panel() -> pd.DataFrame:
    filepath = os.path.join(PROCESSED_DIR, "panel_v1.csv")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    return df


def plot_correlation_matrix(df: pd.DataFrame):
    """
    Correlation across indicators — core evidence for confounding.
    Computes Pearson and Spearman (rank-based) matrices.
    """
    corr_pearson = df[INDICATORS].corr(method="pearson")
    corr_spearman = df[INDICATORS].corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Pearson Correlation")
    sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Spearman (rank) Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_matrix.png"))
    plt.close()

    return corr_pearson, corr_spearman


def flag_outliers(df: pd.DataFrame, mad_threshold: float = DEFAULT_MAD_THRESHOLD) -> pd.DataFrame:
    """
    Robust relative outlier detection using Median Absolute Deviation (MAD).
    Evaluates genuine survey-to-survey changes for forward-filled indicators
    to avoid artificial flat-fill baseline distortions.
    """
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
    outlier_rows = []

    FORWARD_FILLED = {"nutrition_stunting", "literacy_rate"}

    for indicator in INDICATORS:
        is_filled_col = f"{indicator}_is_filled"

        if indicator in FORWARD_FILLED and is_filled_col in df.columns:
            # Compare consecutive GENUINE (non-filled) measurements only
            genuine = df[~df[is_filled_col] & df[indicator].notna()].copy()
            genuine = genuine.sort_values(["country_code", "year"])
            genuine["_change"] = genuine.groupby("country_code")[indicator].diff()

            def calc_mod_z(group):
                vals = group["_change"].dropna()
                if len(vals) < 2:
                  return pd.Series(0.0, index=group.index)
                med = vals.median()
                mad = (vals - med).abs().median()
    # Epsilon check: if MAD is zero or extremely close to 0, return 0.0 z-scores
                if mad < 1e-6 or np.isnan(mad):
                    return pd.Series(0.0, index=group.index)
                return 0.6745 * (group["_change"] - med) / mad

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                genuine["_mod_z"] = genuine.groupby("country_code", group_keys=False).apply(
                    calc_mod_z, include_groups=False
                )

            flagged = genuine[genuine["_mod_z"].abs() > mad_threshold]
            for _, row in flagged.iterrows():
                outlier_rows.append({
                    "country_code": row["country_code"],
                    "country_name": row["country_name"],
                    "year": row["year"],
                    "indicator": indicator,
                    "value": row[indicator],
                    "change": row["_change"],
                    "mod_z_score": round(row["_mod_z"], 2),
                    "note": "Genuine survey-to-survey MAD anomaly",
                })
        else:
            # Densely reported indicators: annual change
            df[f"{indicator}_yoy"] = df.groupby("country_code")[indicator].diff()

            def calc_mod_z_dense(group):
                vals = group[f"{indicator}_yoy"].dropna()
                if len(vals) < 2:
                    return pd.Series(0.0, index=group.index)
                med = vals.median()
                mad = (vals - med).abs().median()
                # Numerical stability check: if MAD is near zero, return 0.0 Z-scores
                if mad < 1e-6 or np.isnan(mad):
                    return pd.Series(0.0, index=group.index)
                return 0.6745 * (group[f"{indicator}_yoy"] - med) / mad

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                df[f"{indicator}_mod_z"] = df.groupby("country_code", group_keys=False).apply(
                    calc_mod_z_dense, include_groups=False
                )

            flagged = df[df[f"{indicator}_mod_z"].abs() > mad_threshold]
            for _, row in flagged.iterrows():
                outlier_rows.append({
                    "country_code": row["country_code"],
                    "country_name": row["country_name"],
                    "year": row["year"],
                    "indicator": indicator,
                    "value": row[indicator],
                    "change": row[f"{indicator}_yoy"],
                    "mod_z_score": round(row[f"{indicator}_mod_z"], 2),
                    "note": "Standard annual MAD anomaly",
                })
            df.drop(columns=[f"{indicator}_yoy", f"{indicator}_mod_z"], inplace=True)

    outliers_df = pd.DataFrame(outlier_rows)

    if not outliers_df.empty:
        overlap_counts = outliers_df.groupby(["country_code", "year"]).size().reset_index(name="n_indicators_flagged")
        outliers_df = outliers_df.merge(overlap_counts, on=["country_code", "year"])
        outliers_df = outliers_df.sort_values("n_indicators_flagged", ascending=False)

    return outliers_df


def check_outlier_threshold_sensitivity(df: pd.DataFrame) -> dict:
    """Sensitivity check across multiple MAD Modified Z-score thresholds."""
    results = {}
    for threshold in [2.5, 3.0, 3.5, 4.0]:
        outliers = flag_outliers(df.copy(), mad_threshold=threshold)
        results[f"MAD > {threshold}"] = len(outliers)
    return results


def plot_time_trends(df: pd.DataFrame, sample_countries: list = None) -> None:
    if sample_countries is None:
        sample_countries = df["country_code"].drop_duplicates().sample(6, random_state=42).tolist()

    fig, axes = plt.subplots(len(INDICATORS), 1, figsize=(10, 14), sharex=True)
    for i, indicator in enumerate(INDICATORS):
        for code in sample_countries:
            subset = df[df["country_code"] == code]
            axes[i].plot(subset["year"], subset[indicator], marker="o", markersize=3, label=code)
        axes[i].set_title(indicator)
        axes[i].legend(fontsize=7, ncol=6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "time_trends_sample.png"))
    plt.close()


def plot_missingness_heatmap(df: pd.DataFrame) -> None:
    missing_flags = pd.DataFrame(index=df.index)
    for indicator in INDICATORS:
        missing_flags[indicator] = df[indicator].isna()

    missing_flags["country_code"] = df["country_code"]
    missing_flags["year"] = df["year"]

    pivot = missing_flags.groupby(["country_code", "year"])[INDICATORS].mean().mean(axis=1).unstack()

    plt.figure(figsize=(14, 20))
    sns.heatmap(pivot, cmap="Reds", cbar_kws={"label": "Fraction of indicators missing"})
    plt.title("Missingness Heatmap (Country x Year)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "missingness_heatmap.png"))
    plt.close()


def check_missingness_mechanism(df: pd.DataFrame) -> dict:
    results = {}
    for indicator in ["nutrition_stunting", "literacy_rate"]:
        coverage = df.groupby("country_code")[indicator].apply(lambda s: s.notna().mean())
        avg_gdp_by_country = df.groupby("country_code")["gdp_per_capita"].mean()

        merged = pd.DataFrame({"coverage": coverage, "avg_gdp": avg_gdp_by_country}).dropna()
        correlation = merged["coverage"].corr(merged["avg_gdp"])

        median_coverage = coverage.median()
        good_coverage_countries = coverage[coverage >= median_coverage].index
        poor_coverage_countries = coverage[coverage < median_coverage].index
        avg_gdp_good = df[df["country_code"].isin(good_coverage_countries)]["gdp_per_capita"].mean()
        avg_gdp_poor = df[df["country_code"].isin(poor_coverage_countries)]["gdp_per_capita"].mean()

        results[indicator] = {
            "coverage_gdp_correlation": round(correlation, 3),
            "avg_gdp_good_coverage_countries": round(avg_gdp_good, 2),
            "avg_gdp_poor_coverage_countries": round(avg_gdp_poor, 2),
            "gap_ratio": round(avg_gdp_good / avg_gdp_poor, 2) if avg_gdp_poor else None,
        }
    return results


def check_outcome_distribution(df: pd.DataFrame) -> dict:
    from scipy.stats import skew

    skew_results = {}
    for indicator in INDICATORS:
        values = df[indicator].dropna()
        skew_results[indicator] = round(skew(values), 3)

    plt.figure(figsize=(12, 8))
    for i, indicator in enumerate(INDICATORS):
        plt.subplot(3, 2, i + 1)
        sns.histplot(df[indicator].dropna(), kde=True)
        plt.title(f"{indicator} (skew={skew_results[indicator]})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "distributions.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df["mortality_u5"].dropna(), kde=True)
    plt.title(f"Mortality (raw), skew={skew_results['mortality_u5']}")
    plt.subplot(1, 2, 2)
    log_mortality = np.log1p(df["mortality_u5"].dropna())
    sns.histplot(log_mortality, kde=True)
    plt.title(f"Mortality (log-transformed), skew={round(skew(log_mortality), 3)}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "mortality_log_comparison.png"))
    plt.close()

    return skew_results


def write_findings(corr_pearson, corr_spearman, missingness_results, outliers_df, skew_results, threshold_sensitivity) -> None:
    lines = ["# EDA Findings — CAUSE-Twin\n"]

    lines.append("## Correlation Matrix — Pearson (evidence for confounding)\n")
    lines.append(corr_pearson.to_markdown())
    lines.append("\n\n## Correlation Matrix — Spearman (robust to skew)\n")
    lines.append(corr_spearman.to_markdown())
    lines.append("\n")

    lines.append("## Missingness Mechanism Check\n")
    for indicator, res in missingness_results.items():
        lines.append(f"**{indicator}**: coverage-GDP correlation = {res['coverage_gdp_correlation']}, "
                     f"good-coverage avg GDP = {res['avg_gdp_good_coverage_countries']}, "
                     f"poor-coverage avg GDP = {res['avg_gdp_poor_coverage_countries']}, "
                     f"gap ratio = {res['gap_ratio']}\n")

    lines.append("\n## Robust MAD Outliers Flagged (report-only, no action taken)\n")
    lines.append(f"Total flagged (at MAD > {DEFAULT_MAD_THRESHOLD}): {len(outliers_df)}\n")
    if not outliers_df.empty:
        multi_overlap = (outliers_df.groupby(["country_code", "year"])["n_indicators_flagged"].first() > 1).sum()
        lines.append(f"Country-years with multi-indicator overlap (n_indicators_flagged > 1): {multi_overlap}\n")
        lines.append(outliers_df.head(30).to_markdown())

    lines.append("\n## Known Data-Quality Notes (Qualitative Reporting Layer Only)\n")
    for (code, year), note in KNOWN_DATA_QUALITY_NOTES.items():
        lines.append(f"- **{code} {year}**: {note}\n")

    lines.append("\n## Outlier Threshold Sensitivity (MAD Metric)\n")
    for threshold_label, count in threshold_sensitivity.items():
        lines.append(f"- {threshold_label}: {count} flagged\n")

    lines.append("\n## Distribution Skew\n")
    for indicator, s in skew_results.items():
        lines.append(f"- {indicator}: skew = {s}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "eda_findings.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"Findings written to {os.path.join(RESULTS_DIR, 'eda_findings.md')}")


def run_eda() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_panel()

    corr_pearson, corr_spearman = plot_correlation_matrix(df)
    plot_time_trends(df)
    plot_missingness_heatmap(df)
    missingness_results = check_missingness_mechanism(df)
    outliers_df = flag_outliers(df)
    skew_results = check_outcome_distribution(df)
    threshold_sensitivity = check_outlier_threshold_sensitivity(df)

    write_findings(corr_pearson, corr_spearman, missingness_results, outliers_df, skew_results, threshold_sensitivity)

    print("\n--- EDA Summary ---")
    print("Pearson correlation:\n", corr_pearson.round(2))
    print("\nSpearman correlation:\n", corr_spearman.round(2))
    print("\nMissingness mechanism check:", missingness_results)
    print(f"\nOutliers flagged (at MAD > {DEFAULT_MAD_THRESHOLD}): {len(outliers_df)}")
    if not outliers_df.empty:
        multi_overlap = (outliers_df.groupby(["country_code", "year"])["n_indicators_flagged"].first() > 1).sum()
        print(f"Country-years with multi-indicator overlap: {multi_overlap}")
    print("\nSkew per indicator:", skew_results)
    print("\nOutlier threshold sensitivity (MAD):", threshold_sensitivity)


if __name__ == "__main__":
    run_eda()