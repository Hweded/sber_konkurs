"""CLI-компилятор презентации из артефактов текущего прогона."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.presentation_builder import build_presentation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("reports/artifacts"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--output", type=Path, default=Path("presentation.pdf"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        build_presentation(args.artifacts, args.figures, args.output)
    except (OSError, ValueError, KeyError):
        logging.exception("Не удалось собрать презентацию")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
