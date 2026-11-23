# music-emotion-disagreement

Analysis code and processed analysis tables for:

> **Disagreement in Music Emotion Judgments: Listener-Related Variation Across Two Datasets**

The study examines whether listener-related variables (musician status / musical training, age) are associated with disagreement in music emotion judgments, using continuous valence and arousal disagreement measures in two non-equivalent datasets:

- **MEET** (Musical Emotion Evaluation Test; primary dataset): 116 short composed stimuli rated by 200 participants (100 musicians, 100 non-musicians). Stimulus-level standard deviation of valence/arousal ratings compared between groups.
- **MERP** (Music Emotion Recognition with Profile information; comparative dataset): 50 full-length songs rated by 259 approved crowdsourced workers. Worker-level mean absolute deviation from song-level consensus compared by training status.

## Data sources

Raw data are **not** redistributed in this repository. Download them from the original public repositories:

| Dataset | Source |
| --- | --- |
| MEET | Mendeley Data: <https://data.mendeley.com/datasets/y2j5ybptzd/3> |
| MERP | Kaggle: <http://www.kaggle.com/kohenyan/music-emotion-recognition-with-profile-information> |

Place the raw files under `data/raw/meet/` and `data/raw/merp/` (mirroring the original file layout). The processed tables under `data/processed/` are included so that all downstream analyses can be verified without re-running the raw-data reconstruction step.

## Software environment

All analyses were conducted in **Python 3.10.21** with the pinned dependencies listed in `requirements.txt`:

| Package | Version |
| --- | --- |
| numpy | 2.0.2 |
| pandas | 2.2.3 |
| scipy | 1.14.1 |
| statsmodels | 0.14.4 |
| matplotlib | 3.9.4 |
| seaborn | 0.13.2 |
| openpyxl | 3.1.5 |

`librosa`/`soundfile` are only required for the optional audio feature-extraction module (`src/music_disagreement/features.py`), which is not part of the analyses reported in the manuscript.

## How to reproduce

### Option 1 — Verify the reported results without raw data

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python verify_results.py
```

`verify_results.py` recomputes the primary statistics reported in the manuscript directly from the processed analysis tables included in this repository (`data/processed/`): the stimulus-level MEET comparisons, the worker-level MERP comparisons, Cohen's d with 95% confidence intervals, and the exploratory two-predictor OLS models. It checks the recomputed values against the reported values and exits with an error if any of them do not match.

### Option 2 — Full pipeline from the raw data

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# place the downloaded MEET files in data/raw/meet/ and MERP files in data/raw/merp/
python run_analysis.py
```

`run_analysis.py` executes the full pipeline in order:

1. `run_step2a()` — data audit, MEET/MERP reconstruction, disagreement summary tables (`data/processed/`)
2. `run_step2b1()` — stimulus-level and worker-level group comparisons, correlations, exploratory OLS models (`outputs/tables/step2b1_*.csv`)
3. effect-size table for the primary comparisons, with 95% confidence intervals (`outputs/tables/effect_sizes_primary_comparisons.csv`)
4. manuscript tables and figures (`outputs/tables/manuscript/`, `outputs/figures/`)

## Repository structure

```
src/music_disagreement/   Analysis package
  audit.py                  raw-data audit utilities
  config.py                 project paths and metric configuration
  disagreement.py           continuous and discrete disagreement metrics
  io_datasets.py            dataset loading helpers
  step2a.py                 data reconstruction and disagreement summaries
  step2b1.py                core listener-factor comparisons and OLS models
  step2b1_5.py              outcome/variable freeze tables
  step2b2_lite.py           cue-conflict readiness check (no audio features)
  stats_models.py           summary statistics, correlations, OLS helpers
  effect_sizes.py           Cohen's d and 95% CIs for primary comparisons
  generate_manuscript_package.py  manuscript tables and figures
  manuscript_plot_style.py  publication plotting style
  features.py               optional audio feature extraction (not used in reported analyses)
  viz.py                    basic plotting helpers
notebooks/                 Exploratory notebooks (audit, disagreement baseline, stats/viz)
data/processed/            Processed analysis tables (see Data sources for raw data)
outputs/tables/            Result tables, including manuscript Table 1 / Table 2 / Supplementary Table S1
```

## Mapping to the manuscript

| Manuscript item | Repository output |
| --- | --- |
| Table 1 (datasets, analytic units, primary measures) | `outputs/tables/manuscript/table1_study_design.csv` |
| Table 2 (primary continuous results) | `outputs/tables/manuscript/table2_main_results.csv` |
| Table 2 note (t, d, 95% CIs) | `outputs/tables/effect_sizes_primary_comparisons.csv` |
| Supplementary Table S1 (secondary/exploratory results) | `outputs/tables/manuscript/supplementary_table_s1_result_placement.csv` |
| Figures 2–4 | `outputs/figures/main/figure1_meet_main_results.*`, `figure2_merp_external_validation.*`, `figure3_cross_dataset_effect_summary.*` |

## Notes

- `data/raw/` is git-ignored and must be populated from the sources above.
- The exploratory OLS models reported in the manuscript use two predictors per dataset (musician-group membership and age in MEET; training status and age in MERP); they correspond to the `model_group_age` rows in `outputs/tables/step2b1_meet_core_results.csv` and the `model_training_age` rows in `outputs/tables/step2b1_merp_validation_results.csv`.
