"""Fail-fast freshness check for generated paper numbers.

Intended use:
  - Run after analysis scripts update CSVs in paper/tables/
  - Run before compiling the LaTeX paper

Exit codes:
  0  OK: generated numbers are present and not older than upstream CSVs
  1  Stale or missing generated outputs
  2  Missing upstream sources (only when --fail-missing-sources is set)

Usage:
  uv run -- python scripts/04_check_paper_numbers_fresh.py

Options:
  --tables-dir paper/tables
  --generated-dir paper/generated
  --fail-missing-sources
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check generated paper numbers are fresh")
    parser.add_argument("--tables-dir", type=Path, default=Path("paper/tables"))
    parser.add_argument("--generated-dir", type=Path, default=Path("paper/generated"))
    parser.add_argument(
        "--fail-missing-sources",
        action="store_true",
        help="Fail if expected upstream CSVs are missing",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    tables_dir: Path = args.tables_dir
    generated_dir: Path = args.generated_dir
    if not tables_dir.is_absolute():
        tables_dir = (repo_root / tables_dir).resolve()
    if not generated_dir.is_absolute():
        generated_dir = (repo_root / generated_dir).resolve()

    sources = [
        tables_dir / "neff_segments.csv",
        tables_dir / "bootstrap_ci_segments.csv",
    ]

    generated_outputs = [
        generated_dir / "paper_numbers.json",
        generated_dir / "paper_numbers.tex",
    ]

    missing_generated = [p for p in generated_outputs if not p.exists()]
    if missing_generated:
        print("Missing generated outputs:")
        for p in missing_generated:
            print(f"- {p.as_posix()}")
        return 1

    missing_sources = [p for p in sources if not p.exists()]
    if missing_sources:
        print("Missing upstream sources:")
        for p in missing_sources:
            print(f"- {p.as_posix()}")
        if args.fail_missing_sources:
            return 2
        # Non-fatal by default: partial workflows are allowed.

    newest_source_mtime = max((_mtime(p) for p in sources if p.exists()), default=None)
    if newest_source_mtime is None:
        print("No upstream sources found; nothing to compare.")
        return 0

    stale = False
    for out in generated_outputs:
        if _mtime(out) < newest_source_mtime:
            stale = True
            print(f"Stale generated output: {out.as_posix()}")

    if stale:
        print("Generated paper numbers are older than upstream CSVs.")
        print("Regenerate with: uv run -- python scripts/03_export_paper_numbers.py")
        return 1

    print("OK: generated paper numbers are fresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
