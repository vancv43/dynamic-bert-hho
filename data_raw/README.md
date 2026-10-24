# Raw CERT r4.2 files

The raw logs are not redistributed here. Download them from the Carnegie Mellon University Software Engineering Institute, https://doi.org/10.1184/R1/12841247.v1 (CC BY 4.0):

1. `answers.tar.bz2`, extracted to `data_raw/answers/answers/` (insider ground truth, scenarios `r4.2-1` to `r4.2-3`).
2. `r4.2.tar.bz2`; place its `http.csv` inside `archive.zip` in the repository root. `02_extract_http_sample.py` streams `http.csv` from that archive without unpacking the 84 GB file.

The processed splits in `data_processed/` are enough to rerun every experiment from `03_build_dataset.py` onward.
