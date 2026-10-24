#!/bin/bash
# Full pipeline in paper order. Place the CERT r4.2 files as described in data_raw/README.md.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
echo '[stage] 01_prepare_malicious.py: Malicious HTTP rows from the CERT answer files'
$PY 01_prepare_malicious.py
echo '[stage] 02_extract_http_sample.py: Reservoir sample of benign HTTP rows'
$PY 02_extract_http_sample.py
echo '[stage] 03_build_dataset.py: Cleaning, deduplication and the original row-level split'
$PY 03_build_dataset.py
echo '[stage] 04_run_baselines.py: TF-IDF and Word2Vec + XGBoost on the original split'
$PY 04_run_baselines.py
echo '[stage] 05_run_hho_search.py: HHO search on the original split, used only by the original-protocol models'
$PY 05_run_hho_search.py
echo '[stage] 06_run_final_models.py: BERT models on the original split'
$PY 06_run_final_models.py
echo '[stage] 07_run_shap.py: SHAP token attributions'
$PY 07_run_shap.py
echo '[stage] 09_build_grouped_dataset.py: User-grouped split'
$PY 09_build_grouped_dataset.py
echo '[stage] 10_build_loso_dataset.py: Leave-one-scenario-out split'
$PY 10_build_loso_dataset.py
echo '[stage] 26_rerun_searches_loso.py: HHO, random search and fusion search on the LOSO training scenarios'
$PY 26_rerun_searches_loso.py
echo '[stage] 19_run_loso_ablation.py: Ten-seed LOSO ablation of nine models'
REVISION_RESULTS=rerun_loso $PY 19_run_loso_ablation.py
echo '[stage] 21_make_figures_v3.py: SHAP figure'
$PY 21_make_figures_v3.py
echo '[stage] 22_verify_protocol_facts.py: Recomputes the split statistics quoted in the paper'
$PY 22_verify_protocol_facts.py
echo '[stage] 23_make_framework_figure.py: Framework figure'
$PY 23_make_framework_figure.py
echo '[stage] 24_make_extra_figures.py: User-audit figure'
$PY 24_make_extra_figures.py
echo '[stage] 27_rs_independent_seeds.py: Five independent random searches and their LOSO runs'
REVISION_RESULTS=rerun_loso $PY 27_rs_independent_seeds.py
echo '[stage] 28_grouped_split_checks.py: Shortcut checks on the user-grouped split'
$PY 28_grouped_split_checks.py
echo '[stage] 29_analyze_final.py: Statistics, intervals, tests and result figures'
$PY 29_analyze_final.py
echo '[stage] all stages complete'
