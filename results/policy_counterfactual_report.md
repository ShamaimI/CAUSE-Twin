# Module 6: Policy Counterfactuals & 2023 Holdout Evaluation Report

## 1. 2023 Holdout Out-of-Sample Predictive Validation
- **Overall Validation Verdict:** `PASSED`
- **Holdout MAE:** `7.829` deaths per 1,000 live births (Pre-committed Target: <= 8.5) | **PASSED**
- **Holdout RMSE:** `11.320`
- **Holdout R²:** `0.828` (Pre-committed Target: >= 0.60) | **PASSED**

*Model evaluated via direct Random Forest nuisance prediction on 2023 level covariates, confirming that out-of-sample predictive validity holds across untouched temporal partitions.*

---

## 2. Global Policy Counterfactuals (Child Lives Saved in 2023)
- **Conservative Target (-2.5% Stunting):** `177,368` child lives saved per year.
- **Moderate Target (-5.0% Stunting):** `354,735` child lives saved per year.
- **Ambitious Target (-10.0% Stunting):** `709,470` child lives saved per year *(Upper-Bound Linear Extrapolation Benchmark — see Warning below)*.

> **Extrapolation & Diminishing Returns Warning:** The -10.0% scenario assumes constant marginal treatment effects across large intervention magnitudes. Because CATEs were estimated around observed local panel variations (~1.0pp), true lives saved under a major 10.0pp global campaign will lie below this linear benchmark if nutrition programs experience diminishing marginal returns at scale.

### Top 5 High-Impact Countries (-5% Target) with Real CBR & 95% CIs
1. **China:** `40,298` lives saved | CBR = `6.4` | CATE = `+0.8941` (95% CI: `[0.6819, 1.1062]`)
2. **Ethiopia:** `26,569` lives saved | CBR = `31.9` | CATE = `+1.2943` (95% CI: `[0.5184, 2.0702]`)
3. **Pakistan:** `26,044` lives saved | CBR = `27.8` | CATE = `+0.7569` (95% CI: `[0.1631, 1.3506]`)
4. **India:** `24,513` lives saved | CBR = `16.1` | CATE = `+0.2111` (95% CI: `[-0.2165, 0.6388]`) *[Contains Zero]*
5. **Congo, Dem. Rep.:** `17,854` lives saved | CBR = `41.3` | CATE = `+0.8172` (95% CI: `[0.1857, 1.4487]`)

---

## 3. Methodological Disclosures & Precision Notes

1. **CBR Ingestion & Data Freshness Protocol:** Annual live births are calculated using authentic per-country World Bank Crude Birth Rate (`SP.DYN.CBRT.IN`) data. Where 2023 values remain uncollected due to reporting lag, values are forward-filled from the most recent country-year observation (2021–2022), matching the Module 2 forward-fill protocol.
2. **India Parameter Uncertainty:** India's CATE point estimate (+0.2111) is non-significant at 95% confidence (CI: [-0.2165, +0.6388]). Its high position in absolute lives saved is an artifact of demographic scale (~24.5M births), not statistical parameter precision.
3. **High-Income Rate Elasticity Mechanism:** High-Income CATE (+0.7411) exceeds Low-Income CATE (+0.6486). One plausible, unverified explanation is healthcare infrastructure efficiency (small nutrition gains translating cleanly into avoided deaths) combined with low baseline floor effects — this project did not test or confirm this mechanism directly. Regardless of cause, >95% of absolute lives saved remain concentrated in Low and Lower-Middle Income nations due to birth volume weighting.
4. **Ethiopia Treatment Efficacy:** Ethiopia exhibits the highest CATE point estimate (+1.2943) with a wide but strictly positive confidence interval (CI: [0.5184, 2.0702]), confirming significant positive treatment efficacy under wide parameter uncertainty.

## 4. Country Export Artifact
Granular country-level projections with CATE standard errors and 95% CIs exported to `results\country_lives_saved_2023.csv`.
