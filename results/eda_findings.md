# EDA Findings — CAUSE-Twin

## Correlation Matrix — Pearson (evidence for confounding)

|                    |   mortality_u5 |   nutrition_stunting |   sanitation_basic |   gdp_per_capita |   literacy_rate |
|:-------------------|---------------:|---------------------:|-------------------:|-----------------:|----------------:|
| mortality_u5       |       1        |             0.684428 |          -0.827063 |        -0.400695 |       -0.789223 |
| nutrition_stunting |       0.684428 |             1        |          -0.768853 |        -0.530465 |       -0.637787 |
| sanitation_basic   |      -0.827063 |            -0.768853 |           1        |         0.448973 |        0.797308 |
| gdp_per_capita     |      -0.400695 |            -0.530465 |           0.448973 |         1        |        0.380059 |
| literacy_rate      |      -0.789223 |            -0.637787 |           0.797308 |         0.380059 |        1        |


## Correlation Matrix — Spearman (robust to skew)

|                    |   mortality_u5 |   nutrition_stunting |   sanitation_basic |   gdp_per_capita |   literacy_rate |
|:-------------------|---------------:|---------------------:|-------------------:|-----------------:|----------------:|
| mortality_u5       |       1        |             0.826049 |          -0.888457 |        -0.87556  |       -0.786418 |
| nutrition_stunting |       0.826049 |             1        |          -0.807969 |        -0.806513 |       -0.687615 |
| sanitation_basic   |      -0.888457 |            -0.807969 |           1        |         0.836374 |        0.744963 |
| gdp_per_capita     |      -0.87556  |            -0.806513 |           0.836374 |         1        |        0.667993 |
| literacy_rate      |      -0.786418 |            -0.687615 |           0.744963 |         0.667993 |        1        |


## Missingness Mechanism Check

**nutrition_stunting**: coverage-GDP correlation = -0.573, good-coverage avg GDP = 5901.51, poor-coverage avg GDP = 27377.34, gap ratio = 0.22

**literacy_rate**: coverage-GDP correlation = -0.499, good-coverage avg GDP = 7049.75, poor-coverage avg GDP = 26679.33, gap ratio = 0.26


## Robust MAD Outliers Flagged (report-only, no action taken)

Total flagged (at MAD > 3.5): 884

Country-years with multi-indicator overlap (n_indicators_flagged > 1): 56

|     | country_code   | country_name                   |   year | indicator        |       value |        change |   mod_z_score | note                        |   n_indicators_flagged |
|----:|:---------------|:-------------------------------|-------:|:-----------------|------------:|--------------:|--------------:|:----------------------------|-----------------------:|
| 115 | SYR            | Syrian Arab Republic           |   2012 | mortality_u5     |    34.2     |    12.6       |         17.54 | Standard annual MAD anomaly |                      3 |
|  72 | MWI            | Malawi                         |   2002 | mortality_u5     |   143.2     |   -95.8       |        -19.14 | Standard annual MAD anomaly |                      3 |
| 611 | SYR            | Syrian Arab Republic           |   2012 | sanitation_basic |    91.6347  |     0.2685    |         -4.23 | Standard annual MAD anomaly |                      3 |
| 803 | SYR            | Syrian Arab Republic           |   2013 | gdp_per_capita   |   985.893   |  -911.768     |         -5.52 | Standard annual MAD anomaly |                      3 |
| 802 | SYR            | Syrian Arab Republic           |   2012 | gdp_per_capita   |  1897.66    | -1054.48      |         -6.32 | Standard annual MAD anomaly |                      3 |
| 460 | MWI            | Malawi                         |   2002 | sanitation_basic |     9.35481 |     0.0017785 |        -28.43 | Standard annual MAD anomaly |                      3 |
| 116 | SYR            | Syrian Arab Republic           |   2013 | mortality_u5     |    36.9     |     2.7       |          4.18 | Standard annual MAD anomaly |                      3 |
| 612 | SYR            | Syrian Arab Republic           |   2013 | sanitation_basic |    91.909   |     0.274345  |         -3.88 | Standard annual MAD anomaly |                      3 |
| 777 | MWI            | Malawi                         |   2002 | gdp_per_capita   |   428.21    |   212.736     |          5.92 | Standard annual MAD anomaly |                      3 |
|  71 | MWI            | Malawi                         |   2001 | mortality_u5     |   239       |    68         |         15.39 | Standard annual MAD anomaly |                      2 |
| 233 | BLR            | Belarus                        |   2002 | sanitation_basic |    91.8772  |    -0.017827  |        -11.48 | Standard annual MAD anomaly |                      2 |
| 811 | TTO            | Trinidad and Tobago            |   2009 | gdp_per_capita   | 14634.1     | -6664.94      |         -3.87 | Standard annual MAD anomaly |                      2 |
| 411 | LAO            | Lao PDR                        |   2022 | sanitation_basic |    85.1292  |     2.03162   |         -4.08 | Standard annual MAD anomaly |                      2 |
|  89 | SDN            | Sudan                          |   2004 | mortality_u5     |   101.1     |     9.2       |         25.18 | Standard annual MAD anomaly |                      2 |
| 427 | MDV            | Maldives                       |   2021 | sanitation_basic |    97.8641  |     0.742737  |         -7.73 | Standard annual MAD anomaly |                      2 |
|  88 | PSE            | West Bank and Gaza             |   2023 | mortality_u5     |    41.9     |    27.8       |        192.91 | Standard annual MAD anomaly |                      2 |
| 769 | MDV            | Maldives                       |   2021 | gdp_per_capita   | 10176.1     |  2782.26      |          5.78 | Standard annual MAD anomaly |                      2 |
| 426 | MDV            | Maldives                       |   2020 | sanitation_basic |    97.1214  |     0.770846  |         -7    | Standard annual MAD anomaly |                      2 |
| 692 | ARM            | Armenia                        |   2022 | gdp_per_capita   |  6571.97    |  1886.79      |          6.56 | Standard annual MAD anomaly |                      2 |
|  85 | PRK            | Korea, Dem. People's Rep.      |   2003 | mortality_u5     |    34.5     |   -63.3       |        -84.04 | Standard annual MAD anomaly |                      2 |
| 425 | MDV            | Maldives                       |   2002 | sanitation_basic |    78.4909  |     1.17705   |          3.54 | Standard annual MAD anomaly |                      2 |
| 232 | BLR            | Belarus                        |   2001 | sanitation_basic |    91.895   |    -0.0186465 |        -11.5  | Standard annual MAD anomaly |                      2 |
| 424 | MDV            | Maldives                       |   2001 | sanitation_basic |    77.3139  |     1.19347   |          3.97 | Standard annual MAD anomaly |                      2 |
| 423 | MDA            | Moldova                        |   2003 | sanitation_basic |    72.4845  |     0.41659   |         -4.77 | Standard annual MAD anomaly |                      2 |
|  82 | PER            | Peru                           |   2001 | mortality_u5     |    35.3     |    -3         |         -3.71 | Standard annual MAD anomaly |                      2 |
| 768 | MDV            | Maldives                       |   2020 | gdp_per_capita   |  7393.89    | -4346.38      |        -12.79 | Standard annual MAD anomaly |                      2 |
| 690 | ALB            | Albania                        |   2023 | gdp_per_capita   |  9740.7     |  1983.74      |          4.68 | Standard annual MAD anomaly |                      2 |
| 698 | BHS            | Bahamas, The                   |   2020 | gdp_per_capita   | 26178.8     | -7461.58      |         -6.81 | Standard annual MAD anomaly |                      2 |
| 822 | VCT            | St. Vincent and the Grenadines |   2020 | gdp_per_capita   |  8351.2     |  -389.391     |         -4.41 | Standard annual MAD anomaly |                      2 |
| 823 | VCT            | St. Vincent and the Grenadines |   2022 | gdp_per_capita   |  9693.56    |  1052.28      |          4.95 | Standard annual MAD anomaly |                      2 |

## Known Data-Quality Notes (Qualitative Reporting Layer Only)

- **OMN 2008**: Plausible Methodological Artifact (Unverified against internal logs): literacy_rate jumped +20.5 points in one year — likely a new census/survey round replacing an older estimate discontinuously.

- **EGY 2021**: Plausible Methodological Artifact (Unverified against internal logs): literacy_rate jumped +10.9 points in one year — likely a survey methodology discontinuity.

- **SYR 2012**: Confirmed Major Shock: mortality_u5 spike (+12.6) coincides with the onset of the Syrian Civil War — real shock, not a data entry error.

- **COD 2010**: Plausible Methodological Artifact: Observed U5MR shift aligns temporally with DHS/MICS survey harmonization windows.

- **LSO 2004**: Plausible Methodological Artifact: Discontinuity corresponds to post-census mortality weighting revisions.


## Outlier Threshold Sensitivity (MAD Metric)

- MAD > 2.5: 1369 flagged

- MAD > 3.0: 1068 flagged

- MAD > 3.5: 884 flagged

- MAD > 4.0: 777 flagged


## Distribution Skew

- mortality_u5: skew = 2.172

- nutrition_stunting: skew = 0.413

- sanitation_basic: skew = -0.999

- gdp_per_capita: skew = 3.21

- literacy_rate: skew = -1.168
