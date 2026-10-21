Anonymous manuscript source

Build: bash scripts/build.sh
Main document: main.tex
The build uses the checked-in numeric tables and vector figure, without private data.

Fig. 1: figures/architecture.pptx is editable native PowerPoint artwork.
The figure uses Arial, native shapes/text/connectors, and enlarged arrowheads.
Regenerate with: python scripts/draw_architecture.py
Dependencies: python-pptx, PyMuPDF, LibreOffice Impress, Liberation Sans (Arial-compatible).
Exports: architecture.pdf (vector), architecture.svg, architecture.png.

Diagram job: show where DEER routing and SBR correction enter the local feature path.
Preserved connections: current features through DEER/base/SBR/output; original radar
bypass to SBR and output; detached type scores to SBR; observation maps to DEER/SBR.
QA: local design and visual review; fixed text boxes with 10 percent width reserve;
large arrowheads; grayscale-safe line styles; no raster content in the PDF.
Paper uses the original IEEEtran conference class at its standard type size.
