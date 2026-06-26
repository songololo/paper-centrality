"""Build script for the ISPRS IJGI (MDPI) submission.

Only main.tex + the local Definitions/ folder (mdpi.cls, mdpi.bst, logos)
live in this directory. All shared content (sections, images, tables, etc.)
is resolved from the parent paper/ directory via TEXINPUTS — no copies, no
symlinks. Mirrors paper/epb_submission/build.py.

Compiles the PDF then creates a self-contained upload zip.

Usage:
    python build.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent  # paper/

# Shared content directories pulled from the parent paper/ directory.
SHARED_DIRS = ["sections", "tables", "generated", "supplementary", "images", "plots"]
SHARED_FILES = ["references.bib"]

# Local directories that must travel with the submission (MDPI class/bst/logos).
LOCAL_DIRS = ["Definitions"]

# Files to exclude from the upload zip.
EXCLUDE_SUFFIXES = {".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".synctex.gz", ".blg", ".json"}
EXCLUDE_NAMES = {".DS_Store"}


def tex_env() -> dict[str, str]:
    """Return env dict with TEXINPUTS/BIBINPUTS/BSTINPUTS pointing at parent paper/."""
    env = os.environ.copy()
    parent = str(PARENT_DIR)
    # "." finds Definitions/mdpi.cls and Definitions/mdpi.bst; parent finds shared content.
    env["TEXINPUTS"] = f".:{parent}:{parent}//:"
    env["BIBINPUTS"] = f".:{parent}:"
    env["BSTINPUTS"] = f".:{parent}:"
    return env


def run_cmd(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    """Run a command in SCRIPT_DIR with the tex env, printing a label."""
    print(f"==> {label}...")
    return subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        env=tex_env(),
        capture_output=True,
        text=True,
    )


def compile_pdf() -> bool:
    """Run pdflatex + bibtex + pdflatex x2. Returns True on success."""
    run_cmd(["pdflatex", "-interaction=nonstopmode", "main.tex"], "pdflatex pass 1")
    result = run_cmd(["bibtex", "main"], "bibtex")
    # Surface bibtex warnings/errors.
    for line in (result.stdout + result.stderr).splitlines():
        if "Warning" in line or "Error" in line:
            print(f"    {line}")
    run_cmd(["pdflatex", "-interaction=nonstopmode", "main.tex"], "pdflatex pass 2")
    run_cmd(["pdflatex", "-interaction=nonstopmode", "main.tex"], "pdflatex pass 3")

    pdf = SCRIPT_DIR / "main.pdf"
    if pdf.exists():
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"==> Success: main.pdf ({size_mb:.1f} MB)")
        return True
    print("==> ERROR: main.pdf was not created. Check main.log for details.")
    return False


def _add_dir(zf: zipfile.ZipFile, src_dir: Path, base: Path) -> None:
    """Add every non-excluded file under src_dir to the zip, relative to base."""
    if not src_dir.exists():
        print(f"    WARNING: {src_dir} not found, skipping")
        return
    for item in src_dir.rglob("*"):
        if (
            item.is_file()
            and item.suffix not in EXCLUDE_SUFFIXES
            and item.name not in EXCLUDE_NAMES
        ):
            zf.write(item, str(item.relative_to(base)))


def create_upload_zip() -> None:
    """Assemble a self-contained zip with all files needed for upload."""
    print("==> Assembling upload zip...")
    date_str = datetime.now().strftime("%Y%m%d")
    zip_name = f"mdpi_submission_{date_str}.zip"
    zip_path = SCRIPT_DIR / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Local top-level files (main.tex, template.tex/pdf, etc.).
        for f in SCRIPT_DIR.iterdir():
            if f.is_file() and f.suffix not in EXCLUDE_SUFFIXES and f.name not in EXCLUDE_NAMES:
                if f.suffix == ".zip":
                    continue  # don't zip ourselves
                zf.write(f, f.name)

        # Local directories that must ship with the submission (Definitions/).
        for dirname in LOCAL_DIRS:
            _add_dir(zf, SCRIPT_DIR / dirname, SCRIPT_DIR)

        # Shared content from the parent paper/ directory.
        for dirname in SHARED_DIRS:
            _add_dir(zf, PARENT_DIR / dirname, PARENT_DIR)

        for fname in SHARED_FILES:
            src = PARENT_DIR / fname
            if src.exists():
                zf.write(src, fname)

    print(f"==> Upload ready: {zip_name}")


def main() -> None:
    os.chdir(SCRIPT_DIR)

    if not compile_pdf():
        sys.exit(1)

    create_upload_zip()
    print("==> Done.")


if __name__ == "__main__":
    main()
