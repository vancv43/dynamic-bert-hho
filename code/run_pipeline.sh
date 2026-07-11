#!/bin/bash
set -e
cd "$(dirname "$0")"
source ../.venv/bin/activate

echo "[pipeline] using pre-downloaded ../archive.zip (Kaggle mirror, streamed via unzip -p)"

echo "[pipeline] STAGE extract_http_sample"
python3 02_extract_http_sample.py

echo "[pipeline] STAGE build_dataset"
python3 03_build_dataset.py

echo "[pipeline] STAGE baselines"
python3 04_run_baselines.py

echo "[pipeline] STAGE hho_search"
python3 05_run_hho_search.py

echo "[pipeline] STAGE final_models"
python3 06_run_final_models.py

echo "[pipeline] STAGE shap"
python3 07_run_shap.py

echo "[pipeline] STAGE figures"
python3 08_make_figures.py

echo "[pipeline] ALL_STAGES_COMPLETE"
