# Processed data splits

Derived from the CERT Insider Threat Test Dataset r4.2, Carnegie Mellon University Software Engineering Institute, https://doi.org/10.1184/R1/12841247.v1, licensed CC BY 4.0. Columns: `text` (cleaned URL and page content), `label` (1 = malicious), `user`, and, in the grouped and LOSO files, twelve request metadata features.

| Protocol | Files | Train | Val. | Test | Malicious test users seen in training |
|---|---|---:|---:|---:|---:|
| Original, row split | `train.csv`, `val.csv`, `test.csv` | 4,545 | 974 | 974 | 43 of 44 |
| User-grouped | `*_grouped.csv` | 4,468 | 1,046 | 979 | 0 of 13 |
| Leave-one-scenario-out | `*_loso.csv` | 5,131 | 901 | 461 | 0 of 30 |
