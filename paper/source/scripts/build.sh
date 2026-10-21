#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
# Recheck the result arithmetic when the local evidence files are available.
# The anonymous source package also builds from its checked-in tables alone.
if [ -f evidence/paper_test_summary.csv ] && [ -f evidence/frozen_selection.json ]; then
  python scripts/generate_tables.py
fi
# The editable PowerPoint figure is authoritative. Rebuild it only when absent.
# Figure regeneration additionally needs python-pptx, PyMuPDF and LibreOffice Impress.
if [ ! -f figures/architecture.pdf ]; then
  python scripts/draw_architecture.py
fi
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build_pass1.log
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build_pass2.log
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build_pass3.log
