# Evidence Routing and Supported Residuals for Cooperative LiDAR–Radar Perception

**Qinao Hu**

School of Communication and Electronic Engineering, East China Normal University, Shanghai, China

[Paper](paper/paper.pdf) · [Pretrained Models](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/tag/v1.0.0) · [Experimental Protocol](#experimental-protocol) · [Citation](#citation)

This repository provides the reference implementation, pretrained models, and reproducible evaluation pipeline for the accompanying manuscript, **Evidence Routing and Supported Residuals for Cooperative LiDAR–Radar Perception**.

## Overview

Missing LiDAR returns and spurious echoes require different responses when constructing features for cooperative perception. The proposed method processes local LiDAR–radar observations before a shared cooperative detection backbone:

- **DEER** predicts independent missing and spurious evidence from traceable deletion/injection supervision. Detached type scores guide feature routing while retaining a positive gain for each available modality.
- **SBR** selects nearby measured radar donors, including a null option, and predicts a residual constrained by observation support and current feature amplitude. Detection supervision learns the correction direction without a feature teacher.

![Architecture of the evidence-routing and supported-residual modules](assets/architecture.png)

*Local feature processing is followed by the common AttFuse cooperative backbone and Vehicle detection head. The architecture figure is reproduced from the accompanying manuscript.*

The implementation includes seven configurations: an internal baseline (**S0**), the individual modules (**A**, **B**), their combination (**AB**), an ordinary gating/residual control (**SUP**), and amplitude/type-prior controls (**N1**, **N2**).

## Main Results

Vehicle **AP3D at IoU = 0.7 (%)** on the V2X-R test split. All configurations use training seed 17 and a 20-epoch budget. Each weather condition contains **3,166 cooperative samples and 30,296 ground-truth boxes**.

| Configuration | Normal | Fog | Snow | Fog/Snow Mean |
|---|---:|---:|---:|---:|
| S0 — internal baseline | 65.46 | 51.25 | 45.67 | 48.46 |
| A — DEER | 64.67 | 51.60 | 46.22 | 48.91 |
| B — SBR | 65.18 | 51.41 | 44.72 | 48.06 |
| AB — DEER + SBR | 65.19 | 51.07 | **49.19** | 50.13 |
| SUP — ordinary gating/residual control | 64.84 | 51.36 | 46.24 | 48.80 |
| N1 — normalized-mean descriptor | 64.37 | 51.03 | 48.69 | 49.86 |
| N2 — multiplicative type prior | **65.87** | **52.52** | 48.73 | **50.62** |

AB improves Snow AP3D@0.7 by **3.52 percentage points** and the Fog/Snow mean by **1.67 points** relative to S0, with small decreases in Normal and Fog. N2 achieves the highest weather mean. These single-seed results characterize the tradeoff between weather conditions within the shared experimental framework.

Full-precision results, AP3D/APBEV at IoU thresholds 0.3/0.5/0.7, checkpoint identities, and experiment manifests are available in [`results/reference/`](results/reference/).

## Installation

### Requirements

- Linux x86_64 and Python 3.10.
- An NVIDIA GPU and a CUDA 12.1 development toolkit, including `nvcc`.
- PyTorch 2.1.2+cu121 and torchvision 0.16.2+cu121.
- Approximately 220 GiB of free disk space for dataset preparation; additional space for checkpoints and predictions.

The reference experiments used RTX 2080 Ti devices reporting approximately 22 GB of memory per GPU. Memory requirements on the standard 11 GB configuration need to be assessed separately.

```bash
sudo apt-get update
sudo apt-get install -y git gh build-essential aria2 p7zip-full libgl1 libglib2.0-0

# Authenticate with an account that can access this private repository.
gh auth login
gh auth setup-git

git clone https://github.com/Hqa77880011/CSCWD-DEER-SBR.git
cd CSCWD-DEER-SBR

conda env create -f environment.yml
conda activate cscwd-repro
bash scripts/install.sh
```

The installation script installs the pinned dependencies, builds the Cython/CUDA extensions, and checks GPU operations. Dependency versions are specified in [`requirements-repro.txt`](requirements-repro.txt) and [`requirements-repro-lock.txt`](requirements-repro-lock.txt).

## Dataset Preparation

The experiments use the authors' V2X-R point clouds at revision [`4c9ff1773b49517464714368d30778b5252bdbd0`](https://huggingface.co/datasets/hx24/V2X-R/tree/4c9ff1773b49517464714368d30778b5252bdbd0). The source data contains **8,083 training**, **829 validation**, and **3,166 test** cooperative samples.

```bash
# Optional: verify access to the pinned dataset revision.
python scripts/prepare_data.py --data-root /path/to/v2xr --probe-only

# Download, verify, extract, and build the fixed ground-truth manifest.
python scripts/prepare_data.py --data-root /path/to/v2xr --workers 2
```

The preparation pipeline verifies source checksums, preserves point clouds and annotations, and records image dimensions for geometric field-of-view filtering. RGB pixels are discarded and are not model inputs. Interrupted downloads can be resumed by repeating the command.

An existing dataset prepared with this project's manifests can be reused by supplying its path through `--data-root`.

## Pretrained Models and Evaluation

The [v1.0.0 release](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/tag/v1.0.0) contains the original selected checkpoints used for the reported results. Each checkpoint is approximately 302 MB and retains its frozen, train-refitted batch-normalization statistics.

| Configuration | Selected Epoch | Checkpoint |
|---|---:|---|
| S0 | 20 | [S0-selected.pt](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/S0-selected.pt) |
| A | 20 | [A-selected.pt](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/A-selected.pt) |
| B | 20 | [B-selected.pt](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/B-selected.pt) |
| AB | 20 | [AB-selected.pt](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/AB-selected.pt) |
| SUP | 20 | [SUP-selected.pt](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/SUP-selected.pt) |
| N1 | 18 | [N1-selected.pt](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/N1-selected.pt) |
| N2 | 18 | [N2-selected.pt](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/N2-selected.pt) |

Shared [evaluation metadata](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/evaluation-metadata.tar.gz), a [release manifest](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/pretrained-release.json), and [SHA256 checksums](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/SHA256SUMS) are also provided. The complete download is approximately 2.02 GiB. Downloads require repository access.

### Evaluate the joint model

After installing the environment and preparing the dataset:

```bash
# Download AB and the shared metadata; verify SHA256 automatically.
python scripts/pretrained.py download --variant AB

# Evaluate the selected AB checkpoint on Normal, Fog, and Snow.
CUDA_VISIBLE_DEVICES=0 python scripts/pretrained.py evaluate \
  --data-root /path/to/v2xr \
  --variant AB --weather all --allow-test-evaluation
```

To reproduce the Snow result alone:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/pretrained.py evaluate \
  --data-root /path/to/v2xr --variant AB --weather snow \
  --output-root outputs/ab_snow --allow-test-evaluation
```

The reference AB/Snow AP3D@0.7 is **49.19382268909672%**.

### Evaluate all configurations

```bash
python scripts/pretrained.py download --variant all
python scripts/pretrained.py verify --variant all

CUDA_VISIBLE_DEVICES=0 python scripts/pretrained.py evaluate \
  --data-root /path/to/v2xr \
  --variant all --weather all --allow-test-evaluation
```

This command runs 21 complete test evaluations sequentially. Predictions, AP3D/APBEV results, and comparisons with the reference AP3D@0.7 are written to:

```text
outputs/pretrained_evaluation/<variant>/<weather>/
  predictions.jsonl
  predictions.jsonl.meta.json
  scores.json.gz
  summary.json
```

`summary.json` reports AP values on a 0–1 scale and `AP3D07_delta_pp` in percentage points. Existing outputs are preserved; use a new `--output-root` for another evaluation. The default exporter uses eight CPU postprocessing workers; set `--postprocess-workers 1` for serial postprocessing.

The evaluation entrypoint supports relocated data paths while verifying the original checkpoint, configuration, scientific source, data-file identities, and frozen BN statistics. Evaluation uses the selected weights directly, with no training or BN refitting. Numerical differences may arise across GPU architectures or toolchains.

## Training from Scratch

Generate the seven experiment configurations and verify the training inputs:

```bash
python scripts/reproduce.py configure --data-root /path/to/v2xr
python scripts/reproduce.py audit
python scripts/reproduce.py preflight --variant all
python scripts/reproduce.py smoke --variant AB
```

Train all configurations from their shared initialization policy:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py train --variant all
```

Evaluate the checkpoints selected during training:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py evaluate \
  --variant all --allow-test-evaluation
```

Use `--variant S0`, `A`, `B`, `AB`, `SUP`, `N1`, or `N2` to run an individual configuration. Resume training with `--resume <path/to/latest.rank0.pt>`. Online resume checkpoints and selected evaluation checkpoints have distinct roles.

## Experimental Protocol

| Component | Setting |
|---|---|
| Training protocol | `teacher_free_v4`, schema 4 |
| Evaluation protocol | `project_fix_v1` |
| Training budget | 20 epochs; 20,200 optimizer updates per configuration |
| Random seed | 17 |
| Precision and batch size | FP32; microbatch 4; accumulation 2; effective batch 8 |
| Training weather | Normal and Fog |
| Model selection | Mean Normal/Fog validation AP3D@0.7 |
| BN evaluation statistics | Refit on 32 fixed training samples; frozen during evaluation |
| Snow | Final testing of frozen models |
| Test postprocessing | Score floor 0.001; rotated BEV NMS IoU 0.15; at most 500 predictions per sample |

Ground truth is fixed across weather conditions, with an ego-frame center ROI of x ∈ [0, 140) m and y ∈ [−40, 40) m and a geometric camera field of view. Ground-truth acceptance is independent of weather-specific point counts. AP uses global confidence ordering and VOC2010 all-points interpolation. The reported weather mean is the arithmetic mean of Fog and Snow AP3D@0.7.

Exact executed configurations are stored in [`repro/configs/`](repro/configs/). The detailed [reproduction guide](docs/REPRODUCING.md) covers checkpoint recovery, data preparation, and protocol checks; this supplementary guide is currently in Chinese.

## Reproducibility Checks

The released evaluation path passed **159 regression tests**. A complete AB/Snow evaluation reproduced all six reported AP3D/APBEV metrics at IoU 0.3/0.5/0.7, and its prediction archive was byte-identical to the original experiment. Verification records are available in [`repro/pretrained_release_verification.json`](repro/pretrained_release_verification.json).

```bash
bash scripts/test.sh

# Include dataset-dependent regression checks.
CSCWD_DATA_ROOT=/path/to/v2xr bash scripts/test.sh

# Check the arithmetic of the stored reference results without inference.
python scripts/reproduce.py results
```

## Repository Structure

```text
assets/              Architecture figure from the manuscript
paper/               Accompanying manuscript PDF
opencood/            Models, datasets, training protocol, and evaluation
project_tools/       Training, prediction export, and diagnostic utilities
scripts/             Installation, data preparation, and reproduction entrypoints
repro/               Executed configurations, manifests, and verification records
results/reference/   Reference metrics and experiment identities
tests/               Module, geometry, protocol, and evaluation regression tests
docs/                Detailed reproduction and validation documentation
CITATION.cff         Machine-readable software and preferred paper citation
```

Raw datasets, local environments, generated predictions, and compiled extensions are excluded from Git. Pretrained checkpoints and the original evaluation metadata are distributed through GitHub Releases.

## Citation

If you use this implementation or its experimental artifacts, please cite the accompanying manuscript:

```bibtex
@unpublished{hu2026evidence,
  author = {Hu, Qinao},
  title  = {Evidence Routing and Supported Residuals for Cooperative {LiDAR}--{Radar} Perception},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://github.com/Hqa77880011/CSCWD-DEER-SBR/blob/main/paper/paper.pdf}
}
```

The citation is also available in [`CITATION.bib`](CITATION.bib) and [`CITATION.cff`](CITATION.cff). The CFF metadata identifies the manuscript as the preferred citation for this repository.

## Acknowledgments and License

This implementation builds on [V2X-R](https://github.com/ylwhxht/V2X-R) and its OpenCOOD-based framework. We acknowledge the upstream authors and the related codebases credited in the [upstream README](readme.md).

Please consult [`NOTICE.md`](NOTICE.md) and the applicable source-file headers for third-party attribution and license terms.
