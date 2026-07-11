# Dynamic BERT-HHO

Code accompanying the paper *"Explainable Deep Learning for Insider Threat Detection: A Metaheuristic Swarm Intelligence Approach Using Harris Hawks Optimization"* (submitted to the *Journal of Data Science and Intelligent Systems*).

This repository contains the data-preparation, modelling, optimization, explainability, and figure-generation scripts used to produce every result reported in the paper. No paper text, no journal template files, and no dataset files are included here — see [Dataset](#dataset) below for how to obtain the data.

## What this study does

- Encodes enterprise HTTP activity logs with **BERT** (contextual embeddings) for binary insider-threat classification.
- Uses **Harris Hawks Optimization (HHO)**, a swarm-intelligence metaheuristic, to search the BERT classifier's hyperparameter space (learning rate, batch size, dropout, weight decay, epochs, number of trainable layers) in place of manual tuning.
- Benchmarks against TF-IDF+XGBoost, Word2Vec+XGBoost, BERT without hyperparameter search, and BERT with manual tuning.
- Applies **SHAP** to the final model for post-hoc explainability.
- Reports results transparently, including a negative finding for HHO on this benchmark (see [Key findings](#key-findings)).

## Dataset

This project uses the **CERT Insider Threat Test Dataset (Scenario 4.2)**, a fully synthetic dataset (no real people, organizations, or PII) created by Carnegie Mellon University's CERT Division with ExactData, LLC, under DARPA I2O sponsorship. It is openly licensed (CC BY 4.0).

Download it from one of:

- **Official source (CMU SEI / Figshare):** https://doi.org/10.1184/R1/12841247.v1 — get `r4.2.tar.bz2` (malicious/background activity logs) and `answers.tar.bz2` (ground-truth red-team incident labels).
- **Kaggle mirror** (faster, includes a pre-merged `http.csv`): https://www.kaggle.com/datasets/mrajaxnp/cert-insider-threat-detection-research

This repo does not redistribute the dataset. Place the downloaded files under `data_raw/` before running the pipeline (see below); the scripts expect either the CMU `.tar.bz2` archives or the Kaggle `archive.zip`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10.13 on Apple Silicon (PyTorch MPS backend); no CUDA GPU is required, though one will be faster.

## Pipeline

Scripts are numbered in execution order (see `code/run_pipeline.sh` for the full orchestration):

| Script | Purpose |
|---|---|
| `01_prepare_malicious.py` | Parses CMU's ground-truth answer files into labelled malicious HTTP rows. |
| `02_extract_http_sample.py` | Streams the (large) `http.csv` out of the archive and reservoir-samples benign rows, without extracting the full file to disk. |
| `03_build_dataset.py` | Cleans text, deduplicates, and creates the stratified train/val/test split. |
| `04_run_baselines.py` | TF-IDF+XGBoost and Word2Vec+XGBoost baselines. |
| `05_run_hho_search.py` | Runs the Harris Hawks Optimization hyperparameter search (`hho.py`) over BERT (`common.py`) on a compute-feasible subsample. |
| `06_run_final_models.py` | Retrains BERT (no HPO), BERT (manual tuning), and the HHO-selected configuration on the full training set; evaluates on the held-out test set. |
| `07_run_shap.py` | SHAP explainability analysis on the final HHO-tuned model. |
| `08_make_figures.py` | Generates every figure/table value reported in the paper from the saved results. |

Run the whole thing with:

```bash
cd code
./run_pipeline.sh
```

`hho.py` is a self-contained NumPy implementation of Harris Hawks Optimization (exploration, soft/hard besiege, Lévy-flight escape) adapted to search BERT hyperparameters; it has no dependency on the rest of the pipeline and can be reused standalone.

## Key findings

Every method reaches near-ceiling performance on this benchmark:

| Model | Accuracy | F1 | Cohen's Kappa | Train time |
|---|---|---|---|---|
| TF-IDF + XGBoost | 1.0000 | 1.0000 | 1.0000 | – |
| Word2Vec + XGBoost | 1.0000 | 1.0000 | 1.0000 | – |
| BERT (no HPO) | 1.0000 | 1.0000 | 1.0000 | 124.9 s |
| BERT (manual tuning) | 1.0000 | 1.0000 | 1.0000 | 134.2 s |
| **Dynamic BERT-HHO** | 0.9990 | 0.9991 | 0.9979 | 172.9 s |

Diagnostic analysis (see the paper's Discussion) traces this ceiling to two properties of the CERT dataset, not to any classifier's capability: the malicious HTTP content field embeds an overt suspicious-keyword vocabulary, and the benign content field is heavily templated (86.8% of a 20,000-row benign sample collapsed into duplicates after cleaning). Under these conditions, HHO-based hyperparameter search did **not** outperform simpler baselines or manual tuning — we report this negative result plainly rather than reframing the comparison. See the paper for the full discussion and recommended fixes (keyword masking / metadata-only features) for a more discriminating benchmark.

## Repository structure

```
dynamic-bert-hho/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── code/
    ├── 01_prepare_malicious.py
    ├── 02_extract_http_sample.py
    ├── 03_build_dataset.py
    ├── 04_run_baselines.py
    ├── 05_run_hho_search.py
    ├── 06_run_final_models.py
    ├── 07_run_shap.py
    ├── 08_make_figures.py
    ├── common.py
    ├── hho.py
    ├── run_pipeline.sh
    └── resume_pipeline.sh
```

## Citation

If you use this code, please cite the paper (details to be updated upon publication):

```bibtex
@article{chau2026dynamicbert-hho,
  title   = {Explainable Deep Learning for Insider Threat Detection: A Metaheuristic Swarm Intelligence Approach Using Harris Hawks Optimization},
  author  = {Chau, Van Van and Meesad, Phayung and Nguyen, Minh Tuan},
  journal = {Journal of Data Science and Intelligent Systems},
  year    = {2026}
}
```

## License

Code released under the MIT License. The CERT dataset is separately licensed (CC BY 4.0) by Carnegie Mellon University and must be obtained from the official sources linked above.
