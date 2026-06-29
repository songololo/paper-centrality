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

# Shared content for the MANUSCRIPT zip. supplementary/ is intentionally excluded:
# the supplement is a SEPARATE deliverable (supplementary.pdf) that MDPI requires to
# be uploaded on its own, not inside the manuscript package.
SHARED_DIRS = ["sections", "tables", "generated", "images", "plots"]
SHARED_FILES = ["references.bib"]

# Local directories that must travel with the submission (MDPI class/bst/logos).
LOCAL_DIRS = ["Definitions"]

# Files to exclude from the upload zip.
EXCLUDE_SUFFIXES = {".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".synctex.gz", ".blg", ".json"}
# build.py is the local build tool; everything else in this dir ships to the journal.
# The supplement is uploaded separately, so its source, PDF, and bbl stay out of the
# manuscript package. (Internal notes / blank MDPI template live in the gitignored
# top-level mdpi/ scratch.)
EXCLUDE_NAMES = {".DS_Store", "build.py", "supplementary.tex", "supplementary.pdf", "supplementary.bbl"}


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


def _pdflatex(stem: str, label: str) -> None:
    run_cmd(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"], label)


def _bibtex(stem: str) -> None:
    result = run_cmd(["bibtex", stem], f"bibtex {stem}")
    for line in (result.stdout + result.stderr).splitlines():
        if "Warning" in line or "Error" in line:
            print(f"    {line}")


def compile_pdfs() -> bool:
    """Compile main.tex and the separate supplementary.tex.

    The two documents cross-reference each other (main -> Figure/Table S#;
    supplement -> main-text equations) through the xr package, so they are
    compiled in interleaved passes to make each other's .aux available.
    """
    _pdflatex("main", "pdflatex main pass 1")  # defines main labels (e.g. eq:gravity)
    _pdflatex("supplementary", "pdflatex supplementary pass 1")  # reads main.aux; defines S-labels
    _bibtex("main")
    _bibtex("supplementary")
    _pdflatex("main", "pdflatex main pass 2")  # reads supplementary.aux + main.bbl
    _pdflatex("supplementary", "pdflatex supplementary pass 2")
    _pdflatex("main", "pdflatex main pass 3")  # settle cross-references
    _pdflatex("supplementary", "pdflatex supplementary pass 3")

    ok = True
    for stem in ("main", "supplementary"):
        pdf = SCRIPT_DIR / f"{stem}.pdf"
        if pdf.exists():
            size_mb = pdf.stat().st_size / (1024 * 1024)
            print(f"==> Success: {stem}.pdf ({size_mb:.1f} MB)")
        else:
            print(f"==> ERROR: {stem}.pdf was not created. Check {stem}.log for details.")
            ok = False

    # Surface only genuine cross-reference problems. The interleaved xr build
    # always reports "multiply defined" for LastPage and for citations shared by
    # both documents — these are harmless artifacts of xr reading the other
    # document's .aux, so they are intentionally NOT flagged here.
    for stem in ("main", "supplementary"):
        log = SCRIPT_DIR / f"{stem}.log"
        if log.exists():
            text_l = log.read_text(errors="ignore").lower()
            if "there were undefined references" in text_l or "there were undefined citations" in text_l:
                print(f"    WARNING ({stem}.log): undefined references/citations — check cross-refs")
    return ok


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

    if not compile_pdfs():
        sys.exit(1)

    create_upload_zip()
    print("==> Separate supplementary deliverable: supplementary.pdf (upload on its own)")
    print("==> Done.")


if __name__ == "__main__":
    main()
