from __future__ import annotations

from .audit import audit_dataset, audit_to_frame
from .config import DATASETS
from .step2a import run_step2a
from .step2b1 import run_step2b1
from .step2b1_5 import run_step2b1_5
from .step2b2_lite import write_step2b2_lite_readiness_note


def ensure_layout() -> dict:
    status = {}
    for name, path in DATASETS.items():
        status[name] = {
            "exists": path.exists(),
            "file_count": len(list(path.rglob("*"))) if path.exists() else 0,
        }
    return status


def run_data_audit() -> dict:
    results = {}
    for name, path in DATASETS.items():
        raw = audit_dataset(path)
        results[name] = {
            "raw": raw,
            "table": audit_to_frame(raw),
        }
    return results


def print_next_steps() -> None:
    print("Project scaffold ready.")
    print("1. Put MEET files in data/raw/meet/")
    print("2. Put MERP files in data/raw/merp/")
    print("3. Start with notebooks/00_data_audit.ipynb or run run_data_audit()")
    print("4. Run run_step2a() after the audit freeze passes")
    print("5. Run run_step2b1() for listener-factor analysis without audio features")
    print("6. Run run_step2b1_5() to freeze outcomes, variable tiers, and writing-line decisions")
    print("7. Run write_step2b2_lite_readiness_note() before any cue-proxy or audio work")


if __name__ == "__main__":
    print_next_steps()
