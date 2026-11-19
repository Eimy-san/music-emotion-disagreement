from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .io_datasets import list_dataset_files, load_tabular, split_files_by_type, standardize_column_names

KEYWORD_GROUPS = {
    "participant": ["participant", "rater", "annotator", "listener", "subject", "user"],
    "stimulus": ["track", "song", "clip", "stimulus", "excerpt", "music", "file"],
    "rating": ["valence", "arousal", "emotion", "label", "rating", "sam"],
    "listener_factor": ["familiar", "liking", "expert", "musician", "genre", "age", "gender"],
}


def detect_candidate_columns(columns: List[str]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {key: [] for key in KEYWORD_GROUPS}
    lowered = [str(col).lower() for col in columns]
    for group, keywords in KEYWORD_GROUPS.items():
        for col, low in zip(columns, lowered):
            if any(keyword in low for keyword in keywords):
                result[group].append(col)
    return result


def audit_dataset(dataset_dir: Path) -> Dict[str, object]:
    files = list_dataset_files(dataset_dir)
    buckets = split_files_by_type(files)
    tabular_summaries = []
    for path in buckets["tabular"]:
        try:
            df = standardize_column_names(load_tabular(path))
            tabular_summaries.append(
                {
                    "file": str(path.name),
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                    "columns": list(df.columns),
                    "candidates": detect_candidate_columns(list(df.columns)),
                }
            )
        except Exception as exc:
            tabular_summaries.append(
                {
                    "file": str(path.name),
                    "rows": None,
                    "cols": None,
                    "columns": [],
                    "candidates": {},
                    "error": str(exc),
                }
            )
    return {
        "dataset_dir": str(dataset_dir),
        "file_count": len(files),
        "tabular_count": len(buckets["tabular"]),
        "audio_count": len(buckets["audio"]),
        "doc_count": len(buckets["docs"]),
        "other_count": len(buckets["other"]),
        "tabular_summaries": tabular_summaries,
    }


def audit_to_frame(audit_result: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for item in audit_result.get("tabular_summaries", []):
        rows.append(
            {
                "file": item.get("file"),
                "rows": item.get("rows"),
                "cols": item.get("cols"),
                "participant_candidates": ", ".join(item.get("candidates", {}).get("participant", [])),
                "stimulus_candidates": ", ".join(item.get("candidates", {}).get("stimulus", [])),
                "rating_candidates": ", ".join(item.get("candidates", {}).get("rating", [])),
                "listener_factor_candidates": ", ".join(item.get("candidates", {}).get("listener_factor", [])),
                "error": item.get("error", ""),
            }
        )
    return pd.DataFrame(rows)

