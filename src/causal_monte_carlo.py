"""
CAUSE-Twin: 50-Seed Monte Carlo & Refutation Stability Module (2023 Holdout Aligned)
Purpose: Stress-test the CausalForestDML estimator across 50 random seed splits 
evaluated on the 2023 holdout test set to match production counterfactual reporting.
"""

import os
import json
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

PANEL_PATH = "data/processed/panel_v1.csv"
RESULTS_DIR = "results"
SUMMARY_OUTPUT = os.path.join(RESULTS_DIR, "monte_carlo_stability_summary.json")

PREDICTORS = ["sanitation_basic", "log_gdp", "literacy_rate", "mortality_shock_pct"]
TREATMENT = "nutrition_stunting"
OUTCOME = "mortality_u5"

def run_monte_carlo_suite(n_seeds=50):
    if not os.path.exists(PANEL_PATH):
        raise FileNotFoundError(f"Cannot find {PANEL_PATH}. Run merge_panel.py first.")

    df = pd.read_csv(PANEL_PATH)
    df["log_gdp"] = np.log1p(df["gdp_per_capita"])
    if "mortality_shock_pct" not in df.columns:
        df["mortality_shock_pct"] = 0.0

    model_vars = [TREATMENT, OUTCOME] + PREDICTORS + ["population", "cbr_real", "country_code", "year", "is_2023_holdout"]
    df_clean = df.dropna(subset=model_vars).copy()

    train_df = df_clean[~df_clean["is_2023_holdout"]].copy()
    test_df = df_clean[df_clean["is_2023_holdout"]].copy()

    # Pre-residualize variables using the exact training split logic
    demean_vars = [TREATMENT, OUTCOME] + PREDICTORS
    for v in demean_vars:
        c_mean = train_df.groupby("country_code")[v].transform("mean")
        y_mean = train_df.groupby("year")[v].transform("mean")
        g_mean = train_df[v].mean()
        
        train_df[f"{v}_res"] = train_df[v] - c_mean - y_mean + g_mean
        
        c_mean_map = train_df.groupby("country_code")[v].mean()
        test_df[f"{v}_res"] = test_df[v] - test_df["country_code"].map(c_mean_map).fillna(g_mean)

    Y_train_res = train_df[f"{OUTCOME}_res"].values
    T_train_res = train_df[f"{TREATMENT}_res"].values
    W_train_res = train_df[[f"{v}_res" for v in PREDICTORS]].values
    X_train_cov = train_df[["log_gdp"]].values

    X_test_cov = test_df[["log_gdp"]].values
    test_weights = test_df["population"] * (test_df["cbr_real"] / 1000.0)

    print("=" * 65)
    print(f" RUNNING 2023 HOLDOUT ALIGNED {n_seeds}-SEED MONTE CARLO SUITE...")
    print("=" * 65)

    monte_carlo_ates = []
    monte_carlo_weighted_ates = []

    for seed in range(n_seeds):
        # Match production hyperparameters: n_estimators=200 (divisible by 4 -> 200), min_samples_leaf=10
        rf_y = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=seed)
        rf_t = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=seed)
        
        dml = CausalForestDML(
            model_y=rf_y, 
            model_t=rf_t, 
            n_estimators=200, 
            min_samples_leaf=10, 
            random_state=seed
        )
        
        dml.fit(Y_train_res, T_train_res, X=X_train_cov, W=W_train_res)
        
        # Evaluate exclusively on the 2023 holdout test set features
        seed_effects = dml.effect(X_test_cov)
        
        unweighted_mean = float(np.mean(seed_effects))
        weighted_mean = float(np.average(seed_effects, weights=test_weights))
        
        monte_carlo_ates.append(unweighted_mean)
        monte_carlo_weighted_ates.append(weighted_mean)
        
        if (seed + 1) % 10 == 0 or seed == 0:
            print(f" -> Completed seed {seed + 1}/{n_seeds} | Unweighted Mean: {unweighted_mean:.4f} | Weighted ATE: {weighted_mean:.4f}")

    ci_unweighted = np.percentile(monte_carlo_ates, [2.5, 97.5])
    ci_weighted = np.percentile(monte_carlo_weighted_ates, [2.5, 97.5])

    summary = {
        "n_seeds": n_seeds,
        "evaluation_subset": "2023_holdout",
        "unweighted_cate_mean": float(np.mean(monte_carlo_ates)),
        "unweighted_ci_lower": float(ci_unweighted[0]),
        "unweighted_ci_upper": float(ci_unweighted[1]),
        "weighted_ate_mean": float(np.mean(monte_carlo_weighted_ates)),
        "weighted_ci_lower": float(ci_weighted[0]),
        "weighted_ci_upper": float(ci_weighted[1]),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SUMMARY_OUTPUT, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 65)
    print(" HOLDOUT-ALIGNED MONTE CARLO RESULTS SUMMARY:")
    print("=" * 65)
    print(f" Unweighted CATE Mean : {summary['unweighted_cate_mean']:.4f} | 95% CI: [{summary['unweighted_ci_lower']:.4f}, {summary['unweighted_ci_upper']:.4f}]")
    print(f" Weighted ATE Mean    : {summary['weighted_ate_mean']:.4f} | 95% CI: [{summary['weighted_ci_lower']:.4f}, {summary['weighted_ci_upper']:.4f}]")
    print(f" Summary exported to  : {SUMMARY_OUTPUT}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_monte_carlo_suite(n_seeds=50)