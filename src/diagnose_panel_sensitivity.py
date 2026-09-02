"""
CAUSE-Twin: Panel Sensitivity Diagnostic Module
Purpose: Stress-test the dataset constructed in merge_panel.py across
three core structural dimensions:
  1. Forward-fill measurement error (Fresh-Only vs. Full-Filled).
  2. Low-variation nutrition sample (Full 217 vs. 142 High-Var Subset).
  3. Non-sovereign territory inclusion (Full 217 vs. 193 Sovereign Subset).

Outputs: Prints comparative regression tables (coef, std err, N) to console 
and exports results/module2_sensitivity_report.md for documentation.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

PANEL_PATH = "data/processed/panel_v1.csv"
REPORT_PATH = "results/module2_sensitivity_report.md"

NON_SOVEREIGN_CODES = {
    "ABW", "ASM", "BMU", "CUW", "CYM", "GIB", "GGL", "GUM", "HKG", "IMN",
    "CHI", "MAC", "MAF", "NCL", "PRI", "PSE", "PYF", "SXM", "TCA", "VIR",
    "VGB", "FLK", "FOA", "MNP"
}

def load_data():
    if not os.path.exists(PANEL_PATH):
        raise FileNotFoundError(f"Cannot find {PANEL_PATH}. Run merge_panel.py first.")
    
    df = pd.read_csv(PANEL_PATH)
    
    # Filter out holdout year
    df = df[~df["is_2023_holdout"]].copy()
    
    # Add sovereign flag if missing
    if "is_sovereign" not in df.columns:
        df["is_sovereign"] = ~df["country_code"].isin(NON_SOVEREIGN_CODES)
        
    df["log_gdp"] = np.log(df["gdp_per_capita"] + 1e-5)
    return df

def run_fe_regression(df, formula, entity_col="country_code"):
    """
    Fits a entity fixed-effects OLS model via within-demeaning.
    Formula example: 'mortality_u5 ~ nutrition_stunting + sanitation_basic + log_gdp'
    """
    vars_in_model = [c.strip() for c in formula.replace("~", "+").split("+")]
    df_clean = df.dropna(subset=vars_in_model + [entity_col]).copy()
    
    # Demean by entity
    for v in vars_in_model:
        df_clean[v] = df_clean[v] - df_clean.groupby(entity_col)[v].transform("mean")
        
    model = smf.ols(formula=formula, data=df_clean).fit()
    
    coef = model.params.get("nutrition_stunting", np.nan)
    se = model.bse.get("nutrition_stunting", np.nan)
    n_obs = int(model.nobs)
    
    return coef, se, n_obs

def execute_diagnostics():
    df = load_data()
    formula = "mortality_u5 ~ nutrition_stunting + sanitation_basic + log_gdp + mortality_shock_pct"
    
    results = []

    # ----------------------------------------------------
    # Check 1: Fresh-Only vs. Full-Filled Nutrition Data
    # ----------------------------------------------------
    c_full, se_full, n_full = run_fe_regression(df, formula)
    results.append({"Check": "1. Data Fill", "Specification": "Full-Filled (Baseline)", "Coef": c_full, "SE": se_full, "N": n_full})

    df_fresh = df[~df["nutrition_stunting_is_filled"]].copy()
    c_fresh, se_fresh, n_fresh = run_fe_regression(df_fresh, formula)
    results.append({"Check": "1. Data Fill", "Specification": "Fresh-Only (Unfilled)", "Coef": c_fresh, "SE": se_fresh, "N": n_fresh})

    df_dist2 = df[df["nutrition_stunting_filled_distance"] <= 2].copy()
    c_dist2, se_dist2, n_dist2 = run_fe_regression(df_dist2, formula)
    results.append({"Check": "1. Data Fill", "Specification": "Distance <= 2 Years", "Coef": c_dist2, "SE": se_dist2, "N": n_dist2})

    # ----------------------------------------------------
    # Check 2: Low-Variation Nutrition Countries Split
    # ----------------------------------------------------
    df_high_var = df[~df["nutrition_stunting_low_variation_country"]].copy()
    c_hvar, se_hvar, n_hvar = run_fe_regression(df_high_var, formula)
    results.append({"Check": "2. Sample Var", "Specification": "Excl. Low-Var (142 Countries)", "Coef": c_hvar, "SE": se_hvar, "N": n_hvar})

    # ----------------------------------------------------
    # Check 3: Sovereign Nations vs. Territories/SARs
    # ----------------------------------------------------
    df_sov = df[df["is_sovereign"]].copy()
    c_sov, se_sov, n_sov = run_fe_regression(df_sov, formula)
    results.append({"Check": "3. Governance", "Specification": "Sovereign-Only (193 Nations)", "Coef": c_sov, "SE": se_sov, "N": n_sov})

    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    
    # Print Console Summary
    print("\n" + "="*70)
    print("      MODULE 2 PANEL SENSITIVITY DIAGNOSTIC REPORT")
    print("="*70)
    for check_name, group in res_df.groupby("Check"):
        print(f"\n--- {check_name} ---")
        for _, row in group.iterrows():
            shift = ((row['Coef'] - c_full) / abs(c_full)) * 100 if row['Specification'] != "Full-Filled (Baseline)" else 0.0
            print(f"  {row['Specification']:<30} | Coef: {row['Coef']:>8.4f} | SE: {row['SE']:>6.4f} | N: {row['N']:>5} | Shift: {shift:>+6.1f}%")
    print("="*70 + "\n")

    # Export Report
    os.makedirs("results", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# Module 2: Panel Sensitivity Diagnostic Report\n\n")
        f.write("| Diagnostic Check | Specification | Stunting Coef | Std Error | Obs (N) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        for _, r in res_df.iterrows():
            f.write(f"| {r['Check']} | {r['Specification']} | {r['Coef']:.4f} | {r['SE']:.4f} | {r['N']} |\n")
        f.write("\n\n*Note: Regressions use Fixed Effects OLS on U5MR with sanitation, log(GDP), and mortality shock controls.*")
    
    print(f"Full report exported to: {REPORT_PATH}")

if __name__ == "__main__":
    execute_diagnostics()

