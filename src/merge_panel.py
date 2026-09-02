"""
CAUSE-Twin: Panel Construction Module (Module 2)
Purpose: Pivot the long-format raw data (from ingest.py) into one wide
country-year panel table, apply forward-fill to sparse predictor
indicators (never to the outcome), flag filled values, the 2023 holdout
year, compute a continuous mortality shock percentage alongside the threshold
anomaly flag, merge total population for weighting, integrate Crude Birth Rate (cbr_real), and flag low-variation countries.
"""

import pandas as pd
import numpy as np
import glob
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

YEAR_START = 2000
YEAR_END = 2023

OUTCOME_INDICATOR = "mortality_u5"          # NEVER forward-filled — it's the outcome variable
POPULATION_INDICATOR = "population"         # Merged for WLS / population-weighting in downstream modules
CBR_INDICATOR = "cbr_real"                  # Crude Birth Rate for population-weighted birth cohorts

FILLABLE_INDICATORS = [                     # predictors — forward-fill applied here only
    "nutrition_stunting",
    "sanitation_basic",
    "gdp_per_capita",
    "literacy_rate",
]


def load_latest_combined() -> pd.DataFrame:
    """Find and load the most recently dated combined_*.csv from data/raw/."""
    files = glob.glob(os.path.join(RAW_DIR, "combined_*.csv"))
    if not files:
        raise FileNotFoundError("No combined_*.csv found in data/raw/. Run ingest.py first.")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading {latest}")
    return pd.read_csv(latest)


def pivot_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format (country, year, indicator, value) into one row per country-year."""
    wide = df_long.pivot_table(
        index=["country_code", "country_name", "year"],
        columns="indicator_name",
        values="value",
        aggfunc="first",  # each (country, year, indicator) should be unique already
    ).reset_index()
    wide.columns.name = None

    # Map raw 'cbr' indicator to 'cbr_real' for downstream consistency
    if "cbr" in wide.columns:
        wide = wide.rename(columns={"cbr": "cbr_real"})
    elif "cbr_real" not in wide.columns:
        wide["cbr_real"] = np.nan

    return wide


def build_full_grid(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every country has one row per year in YEAR_START-YEAR_END,
    even years with no data at all — necessary so forward-fill has
    explicit gaps to fill across, not just gaps pandas never sees.
    """
    countries = wide[["country_code", "country_name"]].drop_duplicates()
    years = pd.DataFrame({"year": range(YEAR_START, YEAR_END + 1)})
    full_index = countries.merge(years, how="cross")

    full = full_index.merge(wide, on=["country_code", "country_name", "year"], how="left")
    return full


def apply_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill sparse predictor indicators within each country (sorted by year).
    No backfill — years before a country's first known value stay missing.
    The outcome variable (mortality_u5), population, and cbr_real are handled specifically.
    """
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

    for col in FILLABLE_INDICATORS:
        if col not in df.columns:
            print(f"WARNING: expected column '{col}' not found — skipping forward fill")
            continue

        was_missing = df[col].isna()

        # A temporary column: the year value if measured this row, else NaN
        df["_temp_measured_year"] = df["year"].where(~was_missing)

        # Forward-fill the indicator value itself
        df[col] = df.groupby("country_code")[col].ffill()

        # Forward-fill the "last measured year" alongside it
        df["_temp_last_measured_year"] = df.groupby("country_code")["_temp_measured_year"].ffill()

        df[f"{col}_is_filled"] = was_missing & df[col].notna()
        df[f"{col}_filled_distance"] = (df["year"] - df["_temp_last_measured_year"]).where(df[col].notna())

        df.drop(columns=["_temp_measured_year", "_temp_last_measured_year"], inplace=True)

    # Impute and forward-fill cbr_real natively
    if "cbr_real" in df.columns:
        df["cbr_real"] = df.groupby("country_code")["cbr_real"].transform(lambda x: x.ffill().bfill())
        global_median = df["cbr_real"].median()
        df["cbr_real"] = df["cbr_real"].fillna(global_median if not np.isnan(global_median) else 18.0)

    return df


def add_holdout_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag 2023 as the held-out test year (see Module 2 decision 4 / Option A)."""
    df["is_2023_holdout"] = df["year"] == 2023
    return df


def add_anomaly_and_shock_metrics(
    df: pd.DataFrame, 
    mortality_col: str = OUTCOME_INDICATOR, 
    sd_threshold: float = 2.5
) -> pd.DataFrame:
    """
    Computes two complementary shock indicators:
      1. `mortality_shock_pct`: Continuous percentage deviation of U5MR from its 
         trailing 5-year moving average. Removes threshold sensitivity and serves as
         a continuous confounder for DAG backdoor adjustment in Module 5.
      2. `anomalous_mortality_year`: Legacy binary Z-score flag (|Z| > 2.5) for threshold checks.
    """
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

    # 1. Trailing 5-year moving average (excluding current year)
    df["_u5mr_5yr_ma"] = df.groupby("country_code")[mortality_col].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=3).mean()
    )

    # 2. Continuous Shock Measure (% deviation from trailing MA)
    df["mortality_shock_pct"] = (df[mortality_col] - df["_u5mr_5yr_ma"]) / df["_u5mr_5yr_ma"]
    df["mortality_shock_pct"] = df["mortality_shock_pct"].fillna(0.0)

    # 3. Legacy YoY Z-Score Anomaly Flag
    df["_yoy_change"] = df.groupby("country_code")[mortality_col].diff()
    change_std = df.groupby("country_code")["_yoy_change"].transform("std")
    change_mean = df.groupby("country_code")["_yoy_change"].transform("mean")
    z_score = (df["_yoy_change"] - change_mean) / change_std

    df["anomalous_mortality_year"] = z_score.abs() > sd_threshold

    # Clean up temporary columns
    df.drop(columns=["_u5mr_5yr_ma", "_yoy_change"], inplace=True)

    return df


def add_low_variation_flag(df: pd.DataFrame, min_real_measurements: int = 2) -> pd.DataFrame:
    """
    Flag countries whose nutrition/literacy data has fewer than
    min_real_measurements genuine (non-filled) values across the whole panel —
    these contribute little/no within-country variation for fixed-effects modeling.
    """
    for col in FILLABLE_INDICATORS:
        is_filled_col = f"{col}_is_filled"
        if is_filled_col not in df.columns:
            continue

        genuine_counts = df.groupby("country_code").apply(
            lambda g: (~g[is_filled_col] & g[col].notna()).sum(),
            include_groups=False
        )
        low_variation_countries = genuine_counts[genuine_counts < min_real_measurements].index

        df[f"{col}_low_variation_country"] = df["country_code"].isin(low_variation_countries)

    return df


def save_processed(df: pd.DataFrame, version: str = "v1") -> str:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filepath = os.path.join(PROCESSED_DIR, f"panel_{version}.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} rows, {df.shape[1]} columns to {filepath}")
    return filepath


def build_panel() -> pd.DataFrame:
    df_long = load_latest_combined()
    wide = pivot_to_wide(df_long)
    full_grid = build_full_grid(wide)
    filled = apply_forward_fill(full_grid)
    flagged = add_holdout_flag(filled)
    flagged = add_anomaly_and_shock_metrics(flagged)
    flagged = add_low_variation_flag(flagged)
    save_processed(flagged)
    return flagged


if __name__ == "__main__":
    panel = build_panel()
    print("\n--- Summary ---")
    print(f"Total rows: {len(panel)}")
    print(f"Countries: {panel['country_code'].nunique()}")
    print(f"Years: {panel['year'].min()}-{panel['year'].max()}")

    print(f"\nNon-null counts per indicator:")
    check_cols = [OUTCOME_INDICATOR, POPULATION_INDICATOR, CBR_INDICATOR] + FILLABLE_INDICATORS
    for col in check_cols:
        if col in panel.columns:
            print(f"  {col}: {panel[col].notna().sum()} / {len(panel)}")
        else:
            print(f"  {col}: NOT PRESENT IN RAW DATA")

    print(f"\nMortality shock summary:")
    print(f"  Anomalous mortality years flagged (binary): {panel['anomalous_mortality_year'].sum()}")
    print(f"  Mean continuous mortality shock (%): {panel['mortality_shock_pct'].mean() * 100:.2f}%")
    print(f"  Max continuous mortality shock (%): {panel['mortality_shock_pct'].max() * 100:.2f}%")

    for col in FILLABLE_INDICATORS:
        flag_col = f"{col}_low_variation_country"
        if flag_col in panel.columns:
            n_countries = panel[panel[flag_col]]["country_code"].nunique()
            print(f"Countries with low variation in {col}: {n_countries}")