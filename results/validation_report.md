# Module 4 — Baseline Validation Report — CAUSE-Twin

## Pre-Committed Model Targets for Module 5/6 DML
- **Linear Baseline Status:** Documented Failure Mode (Out-of-sample linear FE fails to generalize).
- **Benchmark Target for DML Nuisance Models:** Out-of-Sample R² >= 0.60, MAE <= 8.5.

## Multicollinearity (VIF)

- nutrition_stunting: VIF = 3.01

- sanitation_basic: VIF = 3.83

- log_gdp_per_capita: VIF = 3.08

- literacy_rate: VIF = 2.78

- year_trend: VIF = 1.13


## Time-Based K-Fold Results (Full Sample)

|   fold | train_years   | test_years   |   n_train |   n_test |   leakage_row_overlap |   unseen_countries_in_test |   fe_mae_raw |   fe_rmse_raw |   fe_r2_raw |   fe_rmse_log |   fe_r2_log |   rf_mae |   rf_rmse |   rf_r2 |
|-------:|:--------------|:-------------|----------:|---------:|----------------------:|---------------------------:|-------------:|--------------:|------------:|--------------:|------------:|---------:|----------:|--------:|
|      1 | 2000-2009     | 2010-2012    |       732 |      345 |                     0 |                          5 |      161.512 |       162.685 |     -17.859 |        59.659 |      -1.536 |    9.352 |    15.999 |   0.818 |
|      2 | 2000-2012     | 2013-2015    |      1077 |      360 |                     0 |                          4 |       95.084 |        97.801 |      -6.151 |        55.653 |      -1.316 |    7.199 |    13.361 |   0.867 |
|      3 | 2000-2015     | 2016-2018    |      1437 |      372 |                     0 |                          5 |      110.989 |       113.513 |      -8.807 |        52.909 |      -1.131 |    6.892 |    13.495 |   0.861 |
|      4 | 2000-2018     | 2019-2022    |      1809 |      532 |                     0 |                         12 |      118.658 |       121.093 |     -10.624 |        50.201 |      -0.998 |    7.794 |    19.328 |   0.704 |


## Subgroup Bias by Income Tier (Preliminary Linear Diagnostic)

*Note: Subgroup errors will be re-evaluated post-DML in Module 6.*

- High income: Mean Absolute Error = 114.863

- Low income: Mean Absolute Error = 135.037

- Lower middle income: Mean Absolute Error = 120.358

- Upper middle income: Mean Absolute Error = 116.42


## Calibration Diagnostics (Actual = Alpha + Beta * Predicted)

- **Calibration Intercept (Alpha):** `90.765` (Ideal = 0.0)
- **Calibration Slope (Beta):** `0.631` (Ideal = 1.0)
- **Calibration R²:** `0.327`


## Feature Importance Stability Across Folds (Random Forest)

- **nutrition_stunting**: {'mean_importance': 0.063, 'std_importance': 0.003}

- **sanitation_basic**: {'mean_importance': 0.627, 'std_importance': 0.01}

- **log_gdp_per_capita**: {'mean_importance': 0.091, 'std_importance': 0.011}

- **literacy_rate**: {'mean_importance': 0.184, 'std_importance': 0.008}

- **year_trend**: {'mean_importance': 0.035, 'std_importance': 0.008}

- **rank_correlation_first_vs_last_fold**: 1.0


## Robustness Check — Low-Variation Nutrition Exclusion Split

- **Full Sample FE Avg R²:** `-10.860` | **Robust Subset FE Avg R²:** `-10.159`
- **Full Sample RF Avg R²:** `0.812` | **Robust Subset RF Avg R²:** `0.799`
