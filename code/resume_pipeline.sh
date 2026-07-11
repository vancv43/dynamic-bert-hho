#!/bin/bash
set -e
cd "$(dirname "$0")"
source ../.venv/bin/activate

echo "[pipeline] RESUMING from hho_search (dataset/baselines already done)"

echo "[pipeline] STAGE hho_search"
python3 05_run_hho_search.py

echo "[pipeline] STAGE final_models"
python3 06_run_final_models.py

echo "[pipeline] STAGE shap"
python3 07_run_shap.py

echo "[pipeline] STAGE figures"
python3 08_make_figures.py

echo "[pipeline] ALL_STAGES_COMPLETE"
