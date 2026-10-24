<div align="center">

# Dynamic BERT-HHO

### Explainable Deep Learning for Insider Threat Detection: A Metaheuristic Swarm Intelligence Approach Using Harris Hawks Optimization

**Van Van Chau**<sup>1,2</sup> · **Minh Tuan Nguyen**<sup>1</sup> · **Phayung Meesad**<sup>2,*</sup>

<sup>1</sup> Faculty of Information Technology, Posts and Telecommunications Institute of Technology, Ho Chi Minh City, Vietnam<br>
<sup>2</sup> Department of Information Technology Management, King Mongkut's University of Technology North Bangkok, Thailand<br>
<sup>*</sup> Corresponding author

*Artificial Intelligence and Applications*, manuscript #11369 (under review)

![Python](https://img.shields.io/badge/python-3.10-blue) ![License: MIT](https://img.shields.io/badge/code-MIT-green) ![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)

</div>

---

## Contents
[Abstract](#abstract) · [Contributions](#contributions) · [Framework](#framework) · [Evaluation protocols](#evaluation-protocols) · [Results](#results) · [Reproducing the study](#reproducing-the-study) · [Repository layout](#repository-layout) · [Data and license](#data-and-license) · [Citation](#citation)

## Abstract
Insider threats come from trusted users, and their actions hide inside normal activity. This study fine-tunes Bidirectional Encoder Representations from Transformers on the text of web requests, selects its settings with Harris Hawks Optimization, and explains alerts with SHapley Additive exPlanations. A second model fuses the text vector with twelve request metadata features through a fusion head whose width and dropout are also searched. Both models use the web logs of the CERT Insider Threat Test Dataset, release 4.2. An audit of the usual row-level split shows that 43 of 68 malicious training users also appear in the test set and that scenario words separate the classes even without this leakage, so all models score near 100% accuracy there. The models are therefore tested on a leave-one-scenario-out split that holds out an unseen insider scenario, with every search confined to the training scenarios, ten training seeds, bootstrap confidence intervals and paired Wilcoxon tests. Under this split, Harris Hawks tuning improves recall and F1-score over manually chosen settings and matches five equal-budget random searches, while metadata fusion brings no significant gain. The search loss saturates because most configurations classify the validation scenarios perfectly, and configurations with equal loss reach very different recall on the unseen scenario. A baseline of term frequency–inverse document frequency features with eXtreme Gradient Boosting detects every malicious request, which shows that the synthetic text still carries a lexical shortcut. Detection claims on synthetic insider data need scenario-held-out evaluation and a search objective that does not saturate.

## Contributions
1. **Text and metadata fusion detector.** BERT encodes the URL and page text of each HTTP request; Dynamic BERT-HHO-Meta fuses this vector with 12 request metadata features. Backbone and fusion settings are tuned by Harris Hawks Optimization, HHO, and alerts are explained with SHAP.
2. **Leakage audit and corrected protocols.** The standard row-level CERT split leaks incidents: 43 of 44 malicious test users also appear in training. A user-grouped split and a leave-one-scenario-out, LOSO, split remove this leakage.
3. **Ten-seed LOSO ablation.** Default, manual, random-search and HHO settings, three fusion variants and two XGBoost baselines, with bootstrap confidence intervals and Holm-corrected paired Wilcoxon tests. HHO improves on manual settings and matches five equal-budget random searches; metadata fusion brings no significant gain.
4. **The search objective, not the search algorithm, limits tuning.** Configurations with the same zero validation loss reach mean LOSO recall from 0.18 to 0.86.
5. **Timing analysis** of training, inference and search cost for every model on the same hardware.

## Framework
> **Figure 1.** Framework: model building and prediction process. [PDF](figures/fig_framework.pdf)

## Evaluation protocols
| Protocol | Split unit | Train | Val. | Test | Positive rate, test | Malicious test users seen in training |
|---|---|---:|---:|---:|---:|---:|
| Original | Row, stratified | 4,545 | 974 | 974 | 59.4% | 43 of 44 |
| User-grouped | User | 4,468 | 1,046 | 979 | 62.6% | 0 of 13 |
| LOSO | Scenario r4.2-1 held out | 5,131 | 901 | 461 | 15.0% | 0 of 30 |

All searches behind the LOSO results use 2,000 training and 600 validation rows drawn from the LOSO training scenarios, so no test row enters a search. `05_run_hho_search.py` searches on the original split and serves only the original-protocol models.

> **Figure 2.** Rows per user and shared malicious users. [PDF](figures/fig_user_audit.pdf)

## Results
### Original protocol
Every model is at or near 100% accuracy on the row-level split (ROC-AUC 1.000 for all), which the leakage audit traces to incident leakage and to the scenario vocabulary of the generator. The comparison that matters is LOSO.

### Search behaviour
HHO reached zero validation loss in 41 of its 51 evaluations; the five independent random searches in 37 to 42 of 51. Each tuner keeps the first zero-loss configuration it meets.

> **Figure 3.** Validation loss per search evaluation. [PDF](figures/fig_search_loss.pdf)

### LOSO results on the unseen scenario r4.2-1
461 test requests, 69 malicious; threshold 0.5; ten training seeds per model, fifty runs for random search; 95% intervals from a two-level bootstrap over runs and test rows.

| Model | Precision [95% CI] | Recall [95% CI] | F1-score [95% CI] | ROC-AUC |
|---|---:|---:|---:|---:|
| TF-IDF + XGBoost | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 ± 0.000 |
| Word2Vec + XGBoost | 1.000 [1.000, 1.000] | 0.820 [0.721, 0.903] | 0.894 [0.824, 0.948] | 1.000 ± 0.000 |
| BERT, default | 0.371 [0.098, 0.649] | 0.175 [0.007, 0.400] | 0.206 [0.014, 0.435] | 0.969 ± 0.057 |
| BERT, manual | 0.481 [0.190, 0.783] | 0.157 [0.034, 0.301] | 0.219 [0.058, 0.404] | 0.998 ± 0.001 |
| BERT, random search (5 searches) | 0.808 [0.682, 0.895] | 0.476 [0.373, 0.585] | 0.547 [0.441, 0.653] | 0.997 ± 0.003 |
| **Dynamic BERT-HHO** | 0.856 [0.499, 0.989] | 0.514 [0.257, 0.779] | 0.551 [0.285, 0.820] | 0.998 ± 0.001 |
| BERT-Meta, default | 0.572 [0.283, 0.862] | 0.409 [0.131, 0.695] | 0.429 [0.159, 0.701] | 0.973 ± 0.048 |
| BERT-HHO-Meta, default fusion | 0.873 [0.667, 0.993] | 0.468 [0.265, 0.678] | 0.562 [0.359, 0.755] | 0.990 ± 0.025 |
| **Dynamic BERT-HHO-Meta** | 0.872 [0.669, 0.986] | 0.433 [0.246, 0.628] | 0.532 [0.331, 0.723] | 0.982 ± 0.047 |

> **Figure 4.** Per-run LOSO precision, recall and F1-score. Blue: single runs. Red: mean with 95% bootstrap interval. [PDF](figures/fig_loso_ablation.pdf)

### Paired tests
Mean difference, first model minus second, with Holm-adjusted p-value of a two-sided Wilcoxon signed-rank test over ten paired seeds.

| Comparison | ΔPrecision | ΔRecall | ΔF1-score | ΔROC-AUC |
|---|---:|---:|---:|---:|
| BERT-HHO vs. BERT default | +0.485 (p = 0.328) | +0.339 (p = 0.117) | +0.346 (p = 0.176) | +0.029 (p = 0.059) |
| BERT-HHO vs. BERT manual | +0.375 (p = 0.328) | +0.358 (p = 0.023) | +0.332 (p = 0.023) | +0.000 (p = 1.000) |
| BERT-HHO vs. random search | +0.048 (p = 1.000) | +0.038 (p = 1.000) | +0.005 (p = 1.000) | +0.001 (p = 1.000) |
| BERT-HHO-Meta vs. BERT-HHO | +0.015 (p = 1.000) | -0.081 (p = 1.000) | -0.019 (p = 1.000) | -0.016 (p = 1.000) |
| BERT-Meta vs. BERT, both default | +0.201 (p = 1.000) | +0.233 (p = 1.000) | +0.223 (p = 1.000) | +0.004 (p = 1.000) |
| HHO fusion vs. default fusion | -0.001 (p = 1.000) | -0.035 (p = 1.000) | -0.030 (p = 1.000) | -0.008 (p = 1.000) |
| BERT-HHO-Meta vs. TF-IDF + XGBoost | -0.128 (p = 0.027) | -0.567 (p = 0.014) | -0.468 (p = 0.014) | -0.018 (p = 0.014) |

> **Figure 5.** Recall, ROC-AUC and threat scores of the BERT runs. (b) HHO and five random-search configurations all reached zero validation loss. [PDF](figures/fig_seed_scores.pdf)

> **Figure 6.** LOSO confusion matrices, mean ± SD counts over runs. [PDF](figures/fig_confusion.pdf)

### Timing
Mean ± SD over runs, all models timed on the same machine. One HHO search of 51 evaluations took 53 minutes, one random search 56 minutes on average.

| Model | Training, s | Inference, ms per request |
|---|---:|---:|
| TF-IDF + XGBoost | 5.0 ± 1.9 | 0.08 ± 0.01 |
| Word2Vec + XGBoost | 2.4 ± 0.8 | 0.12 ± 0.02 |
| BERT, default | 132.0 ± 9.0 | 8.25 ± 0.73 |
| BERT, manual | 148.1 ± 9.5 | 8.56 ± 0.46 |
| BERT, random search (5 searches) | 133.8 ± 65.9 | 7.66 ± 0.55 |
| **Dynamic BERT-HHO** | 188.1 ± 15.5 | 7.91 ± 0.94 |
| BERT-Meta, default | 134.1 ± 9.2 | 8.37 ± 0.80 |
| BERT-HHO-Meta, default fusion | 190.2 ± 13.0 | 7.77 ± 0.63 |
| **Dynamic BERT-HHO-Meta** | 189.4 ± 12.1 | 7.87 ± 0.86 |

### Explainability
> **Figure 7.** Top ten tokens by mean absolute SHAP value, original-protocol model, 40 test rows. [PDF](figures/fig_shap_summary_v3.pdf)

## Reproducing the study
```bash
pip install -r requirements.txt
cd code
bash run_all.sh            # full pipeline in paper order
python 29_analyze_final.py # statistics, tables and figures from saved runs only
```
`29_analyze_final.py` needs only the files in `results/rerun_loso/`, so every number above can be regenerated in under a minute without retraining. A full rerun of the searches and the ablation retrains every model and takes about a day.

| Script | Purpose |
|---|---|
| `01_prepare_malicious.py` | Malicious HTTP rows from the CERT answer files |
| `02_extract_http_sample.py` | Reservoir sample of benign HTTP rows |
| `03_build_dataset.py` | Cleaning, deduplication and the original row-level split |
| `04_run_baselines.py` | TF-IDF and Word2Vec + XGBoost on the original split |
| `05_run_hho_search.py` | HHO search on the original split, used only by the original-protocol models |
| `06_run_final_models.py` | BERT models on the original split |
| `07_run_shap.py` | SHAP token attributions |
| `09_build_grouped_dataset.py` | User-grouped split |
| `10_build_loso_dataset.py` | Leave-one-scenario-out split |
| `26_rerun_searches_loso.py` | HHO, random search and fusion search on the LOSO training scenarios |
| `19_run_loso_ablation.py` | Ten-seed LOSO ablation of nine models |
| `22_verify_protocol_facts.py` | Recomputes the split statistics quoted in the paper |
| `27_rs_independent_seeds.py` | Five independent random searches and their LOSO runs |
| `28_grouped_split_checks.py` | Shortcut checks on the user-grouped split |
| `29_analyze_final.py` | Statistics, intervals and paired tests |
| `common.py`, `common_meta.py`, `hho.py` | BERT and fusion models, training loop, HHO |

## Repository layout
```
code/              pipeline scripts, run_all.sh
data_processed/    processed splits of CERT r4.2 (CC BY 4.0), see data_processed/README.md
data_raw/          instructions for the raw CERT files (not redistributed)
results/           final_summary.json and all per-run results
  rerun_loso/      search results, rs_seeds/, loso_ablation/ (test probabilities of every run, raw_runs.json)
logs/              run logs of the searches and the ablation
figures/           figures of the paper, vector PDF
```

## Data and license
The study uses the CERT Insider Threat Test Dataset r4.2, Carnegie Mellon University Software Engineering Institute, https://doi.org/10.1184/R1/12841247.v1, licensed CC BY 4.0. The processed splits in this repository are derived from it under the same license. The code is released under the MIT License.

## Citation
The article is under review at *Artificial Intelligence and Applications*. Until it is published, please cite this repository; see `CITATION.cff`.

## Contact
Phayung Meesad, Department of Information Technology Management, King Mongkut's University of Technology North Bangkok, phayung.m@itd.kmutnb.ac.th
