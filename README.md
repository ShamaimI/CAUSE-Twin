```markdown
# CAUSE-Twin: Causal Data Pipeline for Child Mortality Drivers

CAUSE-Twin is a clean, modular, and reproducible causal data pipeline designed to analyze the structural drivers of child mortality. It bridges raw World Bank API ingestion with advanced non-linear causal machine learning (**Double Machine Learning / EconML CausalForestDML**) and global policy counterfactual projections.

---

## 🏗️ Project Architecture & File Directory

```text
Cause-Twin/
│
├── data/
│   ├── raw/                 # Immutable raw API responses from World Bank
│   └── processed/           # Cleaned longitudinal single source of truth (panel_v1.csv)
│
├── results/                 # Generated analytical reports, CSV artifacts, and JSON metrics
│   ├── eda_findings.md
│   ├── validation_report.md
│   ├── canonical_cate_estimates_2023.csv
│   ├── causal_inference_report.md
│   ├── monte_carlo_stability_summary.json
│   ├── shap_values.json
│   └── policy_counterfactual_report.md
│
├── src/                     # Modular execution pipeline (10 Core Scripts)
│   ├── ingest.py
│   ├── merge_panel.py
│   ├── diagnose_panel_sensitivity.py
│   ├── explore.py
│   ├── validate_baseline.py
│   ├── train_canonical_model.py
│   ├── causal_dag.py
│   ├── causal_monte_carlo.py
│   ├── explain.py
│   └── policy_counterfactuals.py
│
└── venv/                    # Isolated Python virtual environment

```

---

## 🔄 End-to-End Execution Flow

To reproduce the analysis from scratch, execute the pipeline modules sequentially:

1. **Data Ingestion (`src/ingest.py`)**
* *Purpose:* Queries the World Bank API to fetch raw country-year indicators (under-5 mortality, stunting, basic sanitation, GDP per capita, literacy rate, population, and crude birth rate) and archives them in `data/raw/`.


2. **Panel Construction & Harmonization (`src/merge_panel.py`)**
* *Purpose:* Transforms raw long-format data into a wide longitudinal panel, builds a complete country-year grid, applies forward-fill imputation strictly to predictor variables, integrates population metrics and real crude birth rates (`cbr_real`), and calculates mortality shocks. Creates the single source of truth at `data/processed/panel_v1.csv`.


3. **Panel Robustness & Sensitivity Audit (`src/diagnose_panel_sensitivity.py`)**
* *Purpose:* Tests whether panel results are artifacts of data cleaning strategies by comparing full-filled baseline data against unfilled data, 2-year observation windows, low-variation country exclusions, and sovereign-only samples (`results/module2_sensitivity_report.md`).


4. **Exploratory Data Analysis (`src/explore.py`)**
* *Purpose:* Analyzes the processed panel to calculate Pearson and Spearman correlation matrices, evaluates missingness mechanisms against GDP gaps, flags anomalies via Median Absolute Deviation (MAD) thresholds, and exports findings to `results/eda_findings.md`.


5. **Baseline Benchmarking & Multicollinearity (`src/validate_baseline.py`)**
* *Purpose:* Computes Variance Inflation Factors (VIF) to rule out multicollinearity and runs time-blocked cross-validation. Empirically proves that traditional linear panel models (`PanelOLS`) fail out-of-sample ($R^2 < 0$), establishing the justification and performance thresholds ($R^2 \ge 0.60$) required for machine learning.


6. **Canonical Model Training (`src/train_canonical_model.py`)**
* *Purpose:* Fits core machine learning models under a 2023 holdout evaluation framework, eliminating drift by setting up downstream consistency artifacts.


7. **Structural Causal Model & DML (`src/causal_dag.py`)**
* *Purpose:* Maps out theoretical backdoor adjustment sets, fits the advanced EconML `CausalForestDML` model across 137 nations, and executes internal refutation tests (placebo treatments, random common causes, and subset splits) to compute Average Treatment Effects (ATE) and heterogeneous treatment effects (CATEs).


8. **Uncertainty Quantification (`src/causal_monte_carlo.py`)**
* *Purpose:* Executes a 50-seed Monte Carlo stability suite aligned with the holdout evaluation framework, certifying robust confidence intervals and exporting the stability summary to `results/monte_carlo_stability_summary.json`.


9. **Explainability Engine (`src/explain.py`)**
* *Purpose:* Computes Shapley Additive exPlanations (SHAP) on top of the fitted models to unpack feature importance and model variance, exporting results to `results/shap_values.json`.


10. **Policy Counterfactual Projections (`src/policy_counterfactuals.py`)**
* *Purpose:* Loads the canonical CATE estimates and real birth rates to simulate country-level and global child lives saved under conservative (-2.5%), moderate (-5.0%), and ambitious (-10.0%) stunting reduction targets, outputting the final executive report to `results/policy_counterfactual_report.md`.



```

```