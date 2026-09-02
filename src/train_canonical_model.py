"""
CAUSE-Twin: Canonical Causal Model Trainer
Purpose: Fits the Single Source of Truth CausalForestDML model on the training panel,
evaluates CATEs exclusively on the N=193 sovereign 2023 holdout, and exports the 
static artifact for downstream modules to consume.
"""

import os
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

# --- Paths ---
PANEL_PATH = "data/processed/panel_v1.csv"
RESULTS_DIR = "results"
CANONICAL_OUTPUT = os.path.join(RESULTS_DIR, "canonical_cate_estimates_2023.csv")

# --- Model Specification (From DAG) ---
PREDICTORS = ["sanitation_basic", "log_gdp", "literacy_rate", "mortality_shock_pct"]
TREATMENT = "nutrition_stunting"
OUTCOME = "mortality_u5"

# Known non-sovereign territories/aggregates to filter out for the N=193 holdout
NON_SOVEREIGN_CODES = [
    "ASM", "ABW", "BMU", "VGB", "CYM", "CUW", "PYF", "GIB", "GUM", 
    "HKG", "MAC", "NCL", "MNP", "PRI", "SXM", "TCA", "VIR", "PSE", 
    "FRO", "GRL", "IMN", "CHI", "KOS", "SMR", "MCO", "AND", "LIE"
]

def train_and_export_canonical_model():
    print("=" * 65)
    print(" TRAINING CANONICAL CAUSAL MODEL (SINGLE SOURCE OF TRUTH)...")
    print("=" * 65)

    if not os.path.exists(PANEL_PATH):
        raise FileNotFoundError(f"Cannot find {PANEL_PATH}. Run merge_panel.py first.")

    df = pd.read_csv(PANEL_PATH)
    
    # 1. Feature Engineering
    df["log_gdp"] = np.log1p(df["gdp_per_capita"])
    if "mortality_shock_pct" not in df.columns:
        df["mortality_shock_pct"] = 0.0

    # Auto-detect the birth rate column name
    cbr_col = "cbr_real" if "cbr_real" in df.columns else "cbr"
    
    model_vars = [TREATMENT, OUTCOME] + PREDICTORS + ["population", cbr_col, "country_code", "year", "is_2023_holdout"]    
    # Critical: Drop NaNs before splitting
    df_clean = df.dropna(subset=model_vars).copy()

    # 2. Split Train vs 2023 Holdout
    train_df = df_clean[~df_clean["is_2023_holdout"]].copy()
    test_df = df_clean[df_clean["is_2023_holdout"]].copy()
    
    # 3. Enforce Sovereign-Only Filter for Evaluation (N=193)
    test_df = test_df[~test_df["country_code"].isin(NON_SOVEREIGN_CODES)].copy()

    print(f" -> Training set size: {len(train_df)} country-years")
    print(f" -> 2023 Holdout set size (Target N=193): {len(test_df)} sovereign nations")

    # 4. TWFE Residualization (Strictly isolated to prevent leakage)
    demean_vars = [TREATMENT, OUTCOME] + PREDICTORS
    for v in demean_vars:
        # Train means
        c_mean = train_df.groupby("country_code")[v].transform("mean")
        y_mean = train_df.groupby("year")[v].transform("mean")
        g_mean = train_df[v].mean()
        train_df[f"{v}_res"] = train_df[v] - c_mean - y_mean + g_mean
        
        # Apply train country means to test set
        c_mean_map = train_df.groupby("country_code")[v].mean()
        test_df[f"{v}_res"] = test_df[v] - test_df["country_code"].map(c_mean_map).fillna(g_mean)

    # 5. Extract Arrays
    Y_train = train_df[f"{OUTCOME}_res"].values
    T_train = train_df[f"{TREATMENT}_res"].values
    W_train = train_df[[f"{v}_res" for v in PREDICTORS]].values
    X_train = train_df[["log_gdp"]].values
    X_test = test_df[["log_gdp"]].values

    # 6. Fit Canonical Model (Production Hyperparameters locked to seed 42)
    print(" -> Fitting CausalForestDML (n_estimators=200, min_samples_leaf=10)...")
    rf_y = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_t = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    
    dml = CausalForestDML(
        model_y=rf_y, 
        model_t=rf_t, 
        n_estimators=200, 
        min_samples_leaf=10, 
        random_state=42
    )
    
    dml.fit(Y_train, T_train, X=X_train, W=W_train)

    # 7. Evaluate on Sovereign Holdout
    print(" -> Generating CATEs for 2023 holdout...")
    cates = dml.effect(X_test)
    cate_lower, cate_upper = dml.effect_interval(X_test, alpha=0.05)

    # 8. Export Canonical Artifact
    test_df["cate"] = cates
    test_df["cate_lower"] = cate_lower
    test_df["cate_upper"] = cate_upper
    
    # Auto-detect birth rate column name
    cbr_col = "cbr_real" if "cbr_real" in test_df.columns else "cbr"
    
    export_cols = ["country_code", "year", "population", cbr_col, "log_gdp", "cate", "cate_lower", "cate_upper"]
    canonical_export = test_df[export_cols].copy()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    canonical_export.to_csv(CANONICAL_OUTPUT, index=False)

    # Calculate final summary check safely
    birth_weights = canonical_export["population"] * (canonical_export[cbr_col] / 1000.0)
    weighted_ate = np.average(canonical_export["cate"], weights=birth_weights)
    unweighted_mean = np.mean(canonical_export["cate"])
    
    print("\n" + "=" * 65)
    print(" CANONICAL ARTIFACT EXPORTED SUCCESSFULLY")
    print("=" * 65)
    print(f" -> Output saved to      : {CANONICAL_OUTPUT}")
    print(f" -> Final N              : {len(canonical_export)} sovereign nations")
    print(f" -> Seed 42 Weighted ATE : {weighted_ate:.4f}")
    print(f" -> Seed 42 Mean CATE    : {unweighted_mean:.4f} (Unweighted)")
    print("=" * 65 + "\n")