"""Build script for Environment and Planning B: SAGE submission.

Only main.tex + SAGE template files live in this directory.
All shared content (sections, images, tables, etc.) is resolved from
the parent paper/ directory via TEXINPUTS — no copies, no symlinks.

Compiles the PDF then creates a self-contained upload zip.

Usage:
    python build.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent  # paper/

# Shared content directories to include in upload zip
SHARED_DIRS = ["sections", "tables", "generated", "supplementary", "images", "plots"]
SHARED_FILES = ["references.bib"]

# Files to exclude from upload zip
EXCLUDE_SUFFIXES = {".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".synctex.gz", ".blg", ".json"}
EXCLUDE_NAMES = {".DS_Store"}


def tex_env() -> dict[str, str]:
    """Return env dict with TEXINPUTS pointing to parent paper/ directory."""
    env = os.environ.copy()
    parent = str(PARENT_DIR)
    env["TEXINPUTS"] = f".:{parent}:{parent}//:"
    env["BIBINPUTS"] = f".:{parent}:"
    env["BSTINPUTS"] = f".:{parent}:"
    return env


def create_placeholder_logo() -> None:
    """Create a blank SAGE_Logo.pdf required by sagej.cls."""
    logo_path = SCRIPT_DIR / "SAGE_Logo.pdf"
    if logo_path.exists():
        return
    print("==> Creating placeholder SAGE_Logo.pdf...")
    with tempfile.TemporaryDirectory() as tmp:
        tex_file = Path(tmp) / "logo.tex"
        tex_file.write_text(
            r"""\documentclass{article}
\usepackage[paperwidth=30mm,paperheight=5mm,margin=0pt]{geometry}
\pagestyle{empty}
\begin{document}
\null
\end{document}
"""
        )
        subprocess.run(
            ["pdflatex", "-interaction=batchmode", "logo.tex"],
            cwd=tmp,
            capture_output=True,
        )
        pdf = Path(tmp) / "logo.pdf"
        if pdf.exists():
            shutil.copy2(pdf, logo_path)
        else:
            print("WARNING: Could not create SAGE_Logo.pdf placeholder")


def run_cmd(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    """Run a command in SCRIPT_DIR with tex env, printing a label."""
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
    # Show bibtex warnings
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
    else:
        print("==> ERROR: main.pdf was not created. Check main.log for details.")
        return False


def create_upload_zip() -> None:
    """Assemble a self-contained zip with all files needed for upload."""
    print("==> Assembling upload zip...")
    date_str = datetime.now().strftime("%Y%m%d")
    zip_name = f"epb_submission_{date_str}.zip"
    zip_path = SCRIPT_DIR / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Local files (main.tex, sagej.cls, SageH.bst, etc.)
        for f in SCRIPT_DIR.iterdir():
            if f.is_file() and f.suffix not in EXCLUDE_SUFFIXES and f.name not in EXCLUDE_NAMES:
                if f.suffix == ".zip":
                    continue  # don't zip ourselves
                zf.write(f, f.name)

        # Shared content from parent paper/ directory
        for dirname in SHARED_DIRS:
            src_dir = PARENT_DIR / dirname
            if not src_dir.exists():
                print(f"    WARNING: {src_dir} not found, skipping")
                continue
            for item in src_dir.rglob("*"):
                if item.is_file() and item.suffix not in EXCLUDE_SUFFIXES and item.name not in EXCLUDE_NAMES:
                    arcname = str(item.relative_to(PARENT_DIR))
                    zf.write(item, arcname)

        for fname in SHARED_FILES:
            src = PARENT_DIR / fname
            if src.exists():
                zf.write(src, fname)

    print(f"==> Upload ready: {zip_name}")


def main() -> None:
    os.chdir(SCRIPT_DIR)
    create_placeholder_logo()

    if not compile_pdf():
        sys.exit(1)

    create_upload_zip()
    print("==> Done.")


if __name__ == "__main__":
    main()
