"""Fail-fast freshness check for generated paper numbers.

Intended use:
  - Run after analysis scripts update CSVs in paper/tables/
  - Run before compiling the LaTeX paper

Exit codes:
  0  OK: generated numbers are present and not older than upstream CSVs
  1  Stale or missing generated outputs

Usage:
  python scripts/04_check_paper_numbers_fresh.py
"""

# %%
from __future__ import annotations

from pathlib import Path


# %%
def _mtime(path: Path) -> float:
    return path.stat().st_mtime


# %%
repo_root = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
tables_dir = (repo_root / "paper/tables").resolve()
generated_dir = (repo_root / "paper/generated").resolve()

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
    raise SystemExit(1)

missing_sources = [p for p in sources if not p.exists()]
if missing_sources:
    print("Missing upstream sources:")
    for p in missing_sources:
        print(f"- {p.as_posix()}")
    # Non-fatal: partial workflows are allowed.

newest_source_mtime = max((_mtime(p) for p in sources if p.exists()), default=None)
if newest_source_mtime is None:
    print("No upstream sources found; nothing to compare.")
else:
    stale = False
    for out in generated_outputs:
        if _mtime(out) < newest_source_mtime:
            stale = True
            print(f"Stale generated output: {out.as_posix()}")

    if stale:
        print("Generated paper numbers are older than upstream CSVs.")
        print("Regenerate with: uv run -- python scripts/03_export_paper_numbers.py")
        raise SystemExit(1)

    print("OK: generated paper numbers are fresh.")
