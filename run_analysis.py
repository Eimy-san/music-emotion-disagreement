from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from music_disagreement.effect_sizes import write_effect_size_table
from music_disagreement.generate_manuscript_package import (
    main as write_manuscript_package,
)
from music_disagreement.step2a import run_step2a
from music_disagreement.step2b1 import run_step2b1


def check_raw_data() -> bool:
    required = [
        PROJECT_ROOT / "data" / "raw" / "meet" / "Raw_Data.xlsx",
    ]
    merp_files = PROJECT_ROOT / "data" / "raw" / "merp"
    merp_present = merp_files.exists() and any(merp_files.glob("*.csv"))
    return all(path.exists() for path in required) and merp_present


def main() -> None:
    if not check_raw_data():
        print(
            "Raw data files were not found under data/raw/meet/ and data/raw/merp/.\n"
            "Download them from the sources listed in README.md, place them in the "
            "expected folders, and re-run this script.\n"
            "To verify the reported results without the raw data, run "
            "`python verify_results.py` instead."
        )
        sys.exit(1)
    run_step2a()
    run_step2b1()
    write_effect_size_table()
    write_manuscript_package()
    print("Full analysis pipeline complete.")


if __name__ == "__main__":
    main()
