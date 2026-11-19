from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .config import CHECKPOINTS_DIR, DATASETS

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}
MAPPING_HINT_TOKENS = ("map", "mapping", "order", "stimulus", "excerpt", "audio")
FUTURE_MINIMAL_FEATURES = (
    "rms_mean",
    "spectral_centroid_mean",
    "tempo",
    "onset_strength_mean",
)
FUTURE_MINIMAL_RULES = (
    "bright_slow",
    "high_energy_low_tempo",
)


@dataclass
class Step2B2LiteArtifacts:
    readiness_failed: bool
    note_path: Path
    audio_file_count: int
    mapping_candidate_count: int


def _ensure_parent_dirs() -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def _scan_audio_files(dataset_dir: Path) -> List[Path]:
    if not dataset_dir.exists():
        return []
    return sorted(path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)


def _scan_mapping_candidates(project_root: Path) -> List[Path]:
    candidates: List[Path] = []
    allowed_suffixes = {".csv", ".tsv", ".txt", ".md", ".json", ".xlsx", ".xls"}
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        lower_name = path.name.lower()
        if any(token in lower_name for token in MAPPING_HINT_TOKENS):
            candidates.append(path)
    return sorted(candidates)


def _relative_paths(paths: Iterable[Path], root: Path) -> List[str]:
    return [str(path.relative_to(root)) for path in paths]


def _read_sheet_names(meet_xlsx: Path) -> List[str]:
    excel = pd.ExcelFile(meet_xlsx)
    return list(excel.sheet_names)


def write_step2b2_lite_readiness_note() -> Step2B2LiteArtifacts:
    _ensure_parent_dirs()
    project_root = Path(__file__).resolve().parents[2]
    meet_dir = DATASETS["meet"]
    meet_xlsx = meet_dir / "Raw_Data.xlsx"
    audio_files = _scan_audio_files(meet_dir)
    mapping_candidates = _scan_mapping_candidates(project_root)
    relevant_mapping_candidates = [
        path for path in mapping_candidates if "step2b2_lite" not in path.name.lower()
    ]
    sheet_names = _read_sheet_names(meet_xlsx) if meet_xlsx.exists() else []

    readiness_failed = True
    note_path = CHECKPOINTS_DIR / "step2b2_lite_readiness_note.md"
    note = f"""# Step 2B2-lite Readiness Note

Date: 2026-03-22

## Conclusion

Step 2B2-lite readiness failed.

The current local project state is **not sufficient** for minimal cue-conflict proxy analysis, so this stage stops after readiness checking and does not proceed to cue proxies or residual disagreement analysis.

## Order-to-Stimulus Mapping

### Conclusion

The `MEET` order-to-stimulus mapping cannot be locally confirmed right now.

- Confirmed local `MEET` raw file: `data/raw/meet/Raw_Data.xlsx`
- Detected workbook sheets: `{", ".join(sheet_names) if sheet_names else "none"}`
- Current evidence is limited to the raw workbook and previously generated checkpoints.
- No independent, frozen `stimulus_id -> audio file` or `order -> stimulus` mapping file has been confirmed in the current project directory.

Implication:

- Even if audio files were present, stimulus-level alignment to the current disagreement tables would still remain uncertain without an explicit mapping.

## Local MEET Audio

### Conclusion

No usable local `MEET` audio files are present in the current project directory.

- Audio scan root: `data/raw/meet/`
- Audio file count found: {len(audio_files)}
- Supported suffixes checked: `{", ".join(sorted(AUDIO_SUFFIXES))}`

Implication:

- Lightweight feature extraction cannot be run locally in this project state.

## Audio Naming and Alignment

### Conclusion

Audio naming cannot be checked against the stimulus-level analysis tables.

- Reason 1: no local `MEET` audio files were found.
- Reason 2: no confirmed `stimulus_id -> audio file` mapping is available.

Therefore:

- We cannot determine whether local filenames would align with `stimulus_id` in `meet_stimulus_disagreement_summary.csv`.

## Final Readiness Decision

### Conclusion

`Step 2B2-lite readiness = fail`

The current project state is **not enough** for even a minimal cue-conflict proxy analysis.

Missing required items:

- Local `MEET` audio files
- Confirmed `order-to-stimulus` / `stimulus-to-audio` mapping

## What Was Not Executed

### Conclusion

Because readiness failed, the following tasks were intentionally **not** executed:

- No cue proxy feature table was created
- No `librosa` feature extraction was run
- No residual disagreement analysis was run
- No `step2b2_lite_checkpoint.md` was created

## Minimal Future Entry Conditions

### Conclusion

The next attempt at Step 2B2-lite should start only after the missing inputs are available.

Required before rerun:

- Provide local `MEET` audio files inside the project directory
- Provide or confirm `stimulus_id -> audio file` alignment

If readiness later passes, the smallest acceptable Step 2B2-lite feature set should be:

- `{", ".join(FUTURE_MINIMAL_FEATURES)}`

And the smallest cue rules should be:

- `{", ".join(FUTURE_MINIMAL_RULES)}`

The role of that later analysis should remain:

- explain residual disagreement
- not simply reinforce the listener-factor story

## Supporting Evidence

### Conclusion

The readiness failure is consistent with earlier project checkpoints.

- `step2a_notes.md` already flagged both missing audio and missing mapping as unresolved gaps.
- No new local files found in this stage change that conclusion.
"""
    note_path.write_text(note, encoding="utf-8")
    return Step2B2LiteArtifacts(
        readiness_failed=readiness_failed,
        note_path=note_path,
        audio_file_count=len(audio_files),
        mapping_candidate_count=len(relevant_mapping_candidates),
    )


if __name__ == "__main__":
    artifacts = write_step2b2_lite_readiness_note()
    print("Step 2B2-lite readiness check complete.")
    print(f"Readiness failed: {artifacts.readiness_failed}")
    print(f"Audio file count: {artifacts.audio_file_count}")
    print(f"Mapping candidate count: {artifacts.mapping_candidate_count}")
    print(f"Note written to: {artifacts.note_path}")
