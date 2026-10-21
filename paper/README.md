# Accompanying Manuscript

**Evidence Routing and Supported Residuals for Cooperative LiDAR–Radar Perception**

**Anonymous Authors** · 2026

[Read the six-page manuscript](paper.pdf) · [LaTeX source](source/) · [Editable Fig. 1](source/figures/architecture.pptx) · [Vector figure](source/figures/architecture.pdf) · [Citation](../CITATION.bib)

The manuscript introduces degradation-specific evidence routing (DEER) and radar-supported, amplitude-bounded residual correction (SBR) before cooperative aggregation. Seven configurations share the V2X-R training and selection protocol. AB improves Snow AP3D@0.7 by 3.52 percentage points; N2 achieves the highest Fog/Snow mean.

The current paper, supporting documents, and citation metadata use anonymous author information while authorship is being confirmed. The scientific title is retained.

## Build the paper

The checked-in source includes the IEEE conference class, all tables, and the vector figure. It builds without model inference or private experiment records:

```bash
cd paper/source
bash scripts/build.sh
```

The output is `main.pdf`. The source also imports directly into Overleaf with `main.tex` as the main document.

To regenerate Fig. 1, install `python-pptx`, `PyMuPDF`, LibreOffice Impress, and Liberation Sans, then run `python scripts/draw_architecture.py` from the source directory. The PowerPoint uses editable native shapes, text, and connectors; the manuscript embeds its vector PDF export.

[Supporting documents](supporting/) include the anonymous title page, abstract, highlights, and cover-letter draft. Their inclusion here does not imply that the conference requires them.
