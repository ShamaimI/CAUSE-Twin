"""
Digital Twin: SHAP Explainability Engine (Module 7)
Computes global feature importances and per-country SHAP attributions.
"""

import os
import json
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor

PANEL_PATH = "data/processed/panel_v1.csv"
RESULTS_DIR = "results"
SHAP_OUTPUT_PATH = os.path.join(RESULTS_DIR, "shap_values.json")

PREDICTORS = ["sanitation_basic", "log_gdp", "literacy_rate", "mortality_shock_pct", "nutrition_stunting"]
OUTCOME = "mortality_u5"


def run_explainability():
    if not os.path.exists(PANEL_PATH):
        raise FileNotFoundError(f"Cannot find {PANEL_PATH}.")

    df = pd.read_csv(PANEL_PATH)
    df["log_gdp"] = np.log1p(df["gdp_per_capita"])
    if "mortality_shock_pct" not in df.columns:
        df["mortality_shock_pct"] = 0.0

    df_clean = df.dropna(subset=PREDICTORS + [OUTCOME]).copy()

    X = df_clean[PREDICTORS]
    y = df_clean[OUTCOME]

    # Fit Random Forest Baseline
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X, y)

    # Calculate SHAP Values
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # Compute Mean Absolute SHAP Importance per Feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = dict(zip(PREDICTORS, [float(v) for v in mean_shap]))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SHAP_OUTPUT_PATH, "w") as f:
        json.dump(feature_importance, f, indent=4)

    print("=" * 75)
    print("      MODULE 7: SHAP EXPLAINABILITY ENGINE COMPLETE")
    print("=" * 75)
    print("\nGlobal Feature Importance (Mean |SHAP Value|):")
    for feat, val in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {feat:<22}: {val:.4f}")

    print(f"\nSHAP values exported to: {SHAP_OUTPUT_PATH}\n" + "=" * 75)


if __name__ == "__main__":
    run_explainability()