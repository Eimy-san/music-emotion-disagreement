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


def main() -> None:
    run_step2a()
    run_step2b1()
    write_effect_size_table()
    write_manuscript_package()
    print("Full analysis pipeline complete.")


if __name__ == "__main__":
    main()
