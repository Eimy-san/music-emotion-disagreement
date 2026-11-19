from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINTS_DIR = REPORTS_DIR / "checkpoints"

DATASETS = {
    "meet": RAW_DIR / "meet",
    "merp": RAW_DIR / "merp",
}

PRIMARY_DISAGREEMENT_METRIC = "std"

CONTINUOUS_DISAGREEMENT_METRICS = ("std", "mad", "iqr")
DISCRETE_DISAGREEMENT_METRICS = ("entropy", "top_label_proportion")

DEFAULT_PLOT_STYLE = {
    "style": "whitegrid",
    "palette": "deep",
    "context": "talk",
}

CUE_CONFLICT_RULES = {
    "bright_slow": {
        "all_of": ["high_spectral_centroid", "low_tempo"],
        "description": "Bright timbre with slow pulse.",
    },
    "fast_minor_proxy": {
        "all_of": ["high_tempo", "minor_mode_proxy"],
        "description": "Fast pulse with minor or low-valence tonal cue.",
    },
    "stable_harmony_high_flux": {
        "all_of": ["stable_harmony_proxy", "high_spectral_flux"],
        "description": "Stable harmony with rapidly changing surface energy.",
    },
}

