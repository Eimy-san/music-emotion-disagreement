from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".json"}


def list_dataset_files(dataset_dir: Path) -> List[Path]:
    if not dataset_dir.exists():
        return []
    return sorted(p for p in dataset_dir.rglob("*") if p.is_file())


def split_files_by_type(files: Iterable[Path]) -> Dict[str, List[Path]]:
    buckets = {"tabular": [], "audio": [], "docs": [], "other": []}
    for path in files:
        suffix = path.suffix.lower()
        if suffix in TABULAR_SUFFIXES:
            buckets["tabular"].append(path)
        elif suffix in {".wav", ".mp3", ".flac", ".ogg"}:
            buckets["audio"].append(path)
        elif suffix in {".md", ".txt", ".pdf", ".html"}:
            buckets["docs"].append(path)
        else:
            buckets["other"].append(path)
    return buckets


def load_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return pd.json_normalize(payload)
    raise ValueError(f"Unsupported tabular file: {path}")


def preview_tabular_files(dataset_dir: Path, limit: int = 3) -> Dict[str, pd.DataFrame]:
    files = split_files_by_type(list_dataset_files(dataset_dir))["tabular"][:limit]
    previews: Dict[str, pd.DataFrame] = {}
    for path in files:
        try:
            previews[path.name] = load_tabular(path).head()
        except Exception:
            continue
    return previews


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        col: (
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )
        for col in df.columns
    }
    return df.rename(columns=renamed)

