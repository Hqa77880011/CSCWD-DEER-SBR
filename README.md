# Evidence Routing and Supported Residuals for Cooperative LiDAR–Radar Perception

**Anonymous Authors**

[Paper](paper/paper.pdf) · [LaTeX source](paper/source/) · [Pretrained Models](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/tag/v1.0.0) · [Reference Protocol](#experimental-protocol) · [Supplemental Experiments](#supplemental-experiments) · [Server Training Commands](#server-training-commands) · [Citation](#citation)

This repository provides the reference implementation, pretrained models, and reproducible evaluation pipeline for the accompanying manuscript, **Evidence Routing and Supported Residuals for Cooperative LiDAR–Radar Perception**.

The original seven-model release uses `teacher_free_v4` (schema 4). Supplemental experiments use `supplement_v1` (schema 5), with independent configurations and output directories. The AP results and released checkpoints below belong to the original experiments; the supplemental implementation has passed local functional checks and awaits real-data training and evaluation.

Documentation: [Reproduction guide](docs/REPRODUCING.md) · [Supplemental experiment plan and implementation](docs/13_补充实验方案与实施说明.md) · [Validation records](docs/VALIDATION.md). These detailed guides are currently in Chinese.

## Overview

Missing LiDAR returns and spurious echoes require different responses when constructing features for cooperative perception. The proposed method processes local LiDAR–radar observations before a shared cooperative detection backbone:

- **DEER** predicts independent missing and spurious evidence from traceable deletion/injection supervision. Detached type scores guide feature routing while retaining a positive gain for each available modality.
- **SBR** selects nearby measured radar donors, including a null option, and predicts a residual constrained by observation support and current feature amplitude. Detection supervision learns the correction direction without a feature teacher.

![Architecture of the evidence-routing and supported-residual modules](assets/architecture.png)

*Local feature processing is followed by the common AttFuse cooperative backbone and Vehicle detection head. The architecture figure is available as an [editable PowerPoint](paper/source/figures/architecture.pptx) and [vector PDF](paper/source/figures/architecture.pdf).*

The reference release includes seven configurations: an internal baseline (**S0**), the individual modules (**A**, **B**), their combination (**AB**), an ordinary gating/residual control (**SUP**), and amplitude/type-prior controls (**N1**, **N2**). The supplemental implementation adds mechanism controls, external-method comparisons, additional data protocols, and a second cooperative backbone.

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

AB improves Snow AP3D@0.7 by **3.52 percentage points** and the Fog/Snow mean by **1.67 points** relative to S0, with small decreases in Normal and Fog. N2 achieves the highest weather mean. Snow was excluded from training and model selection in this reference batch. Its published test scores are excluded from subsequent structure-selection decisions.

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

Exact executed configurations are stored in [`repro/configs/`](repro/configs/). The original reproduction and pretrained-evaluation entrypoints load the frozen scientific source from [`repro/archive/legacy_v4_runtime.zip`](repro/archive/legacy_v4_runtime.zip), preserving the original configuration and checkpoint identities. The [reproduction guide](docs/REPRODUCING.md) covers checkpoint recovery, data preparation, and protocol checks.

## Supplemental Experiments

[`scripts/supplement.py`](scripts/supplement.py) provides the supplemental CLI and calls the existing project runner. The experiment matrix is defined in [`cscwd_configs/supplement/matrix.json`](cscwd_configs/supplement/matrix.json); dataset templates are available for [V2X-R](cscwd_configs/supplement/v2xr.yaml), [V2X-Radar-C](cscwd_configs/supplement/v2xradar.yaml), and [K-Radar](cscwd_configs/supplement/kradar.yaml).

| Experiment block | Implemented scope |
|---|---|
| Training budgets and repeats | Independent 20/30/60/90-epoch recipes with matching learning-rate schedules; seeds 17/29/43; actual updates, losses, learning rates, development scores, and selected epochs |
| Mechanism controls | S0/A/B/AB/SUP/N1/N2, M1–M6, NO_NULL, residual bound `rho`, support radius, and inference-time signal interventions |
| External methods and second backbone | AttFuse, AttFuse+MDD, L4DR, CoBEVT, and Where2comm; paired Where2comm baseline/+F with shared common-parameter initialization |
| Data protocols | V2X-R fixed GT; V2X-Radar-C vehicle ego with its real roadside partner; K-Radar as a single-vehicle input, with dataset-specific coordinates, classes, ROI, and evaluators |
| Robustness and diagnostics | Deletion, injection, coexistence, radar retention, missing modalities, heterogeneous agents, ego-only/agent count, localization, and asynchronous inputs; type, routing, donor, support, and residual-amplitude diagnostics |
| Statistical and efficiency analysis | Paired-seed summaries, within-seed weather means, budget comparisons, sequence bootstrap with AP recomputation, and CUDA-synchronized timing with explicit processing boundaries |

`F` denotes one AB or N2 structure frozen from the complete S0/AB/N2 development runs for all three seeds. It has **not yet been selected**. Development selection, training-only BN handling, and final testing are separate; `latest.rank0.pt`, `selected.pt`, and `budget_epochNNN.pt` retain distinct roles.

External comparisons retain their method-specific modules, objectives, and registered native candidate recipes. L4DR is explicitly a four-channel adaptation with FAD, foreground supervision, bidirectional MME, and MSGF under the project's detection protocol. MDD requires a verified clean training reference. HRCP and REL remain blocked by insufficient verifiable implementation material and have no executable training configurations.

After installing the environment, inspect the CLI or run the synthetic CPU check without a dataset or pretrained weights:

```bash
python scripts/supplement.py --help
python scripts/supplement.py smoke
```

The smoke check performs synthetic forward, backward, and inference passes with zero optimizer updates and produces no AP results.

<a id="server-training-commands"></a>

### 服务器分阶段训练命令

以下命令用于**已安装项目环境、已准备数据的 Linux 服务器**，按阶段复制执行。先修改服务器路径，再在同一 Bash 会话运行；每个 `run-suite --execute` 都会启动真实训练。同一个 suite 内顺序运行，已成功完成且配置完全一致的先导由 `--skip-completed` 复用。此次 README 更新只校验命令，没有连接实验服务器或启动训练。

阶段顺序：环境预检 → 可选执行配置测量 → S0/AB/N2 seed17 → SUP seed17 → 根据开发曲线冻结预算 → 内部/外部三种子 → 冻结 F → M1/M3/M4 → 按研究问题执行扩展。外部方法可在独立 GPU 上与内部阶段并行；依赖 F 的阶段必须等待三种子开发证据齐全。

#### 阶段 0：设置服务器路径和预检函数

`RUNS` 使用新的补充实验批次目录。`CUDA_VISIBLE_DEVICES` 选择物理 GPU，进程内统一使用 `DEVICE=cuda:0`。`EPOCHS=60` 是先导候选；内部组保持 FP32、microbatch4、累积2、有效batch8。外部方法保留原生 batch 和 schedule。

```bash
# 按服务器实际位置修改这三项；不要填本地 Windows 路径。
export REPO=/data/projects/CSCWD-DEER-SBR
export DATA=/data/v2xr
export RUNS=/data/runs/supplement_v1_20261024

cd "$REPO"
conda activate cscwd-repro
set -euo pipefail
export PYTHONUTF8=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
DEVICE=cuda:0
EPOCHS=60
PLANS="$RUNS/plans"
CHECKS="$RUNS/checks"
LOGS="$RUNS/logs"
REPORTS="$RUNS/reports"
FINAL_MODEL="$RUNS/frozen_final.json"
EXECUTION_ARGS=()  # 测量并冻结执行配置后，设为 (--execution-config "$WINNER")
mkdir -p "$PLANS" "$CHECKS" "$LOGS" "$REPORTS"
git rev-parse HEAD | tee "$CHECKS/repository_commit.txt"
```

每阶段先 `plan --write-configs`，再 `check_plan`，最后训练。下面的函数检查预检报告中的路径和 `formal_blockers`，并读取两个真实 train 样本；单看 `preflight` 的退出码不足以判断能否训练。

```bash
check_config() {
  local config="$1"
  local run_id
  run_id="$(basename "$config" .yaml)"
  python scripts/supplement.py preflight --config "$config" \
    > "$CHECKS/preflight_${run_id}.json"
  python - "$CHECKS/preflight_${run_id}.json" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
missing = [k for k, v in report['paths'].items() if not v['exists']]
if report['formal_blockers'] or missing:
    raise SystemExit(f"BLOCKED: {report['formal_blockers']}; missing paths: {missing}")
print('PASS preflight:', sys.argv[1])
PY
  python scripts/supplement.py check-data --config "$config" --samples 2 \
    | tee "$CHECKS/input_${run_id}.log"
}

check_plan() {
  local config
  while IFS= read -r config; do
    check_config "$config"
  done < <(python -c \
    'import json,sys; print("\n".join(t["config"] for t in json.load(open(sys.argv[1]))["tasks"]))' \
    "$1")
}
```

#### 阶段 1：环境和数据预检

```bash
nvidia-smi
python -m pip check

# 依赖/设备导入预检，无下载、训练或数据读取。
python scripts/supplement.py environment --native-v2xr --device "$DEVICE" \
  --output "$CHECKS/environment.json"

# CUDA 算子、spconv、合成基线及 checkpoint 检查。
python project_tools/check_environment.py --output "$CHECKS/environment_gpu.json" \
  | tee "$LOGS/environment_gpu.log"

# 21 路 CPU 合成前向/反向/推理；零优化更新，无 AP。
python scripts/supplement.py smoke --output "$CHECKS/synthetic_smoke.json"
```

通过条件：依赖导入报告为 `ready`，GPU 合成环境检查为 `passed`；后续各配置 `formal_blockers=[]` 且 train 样本可读。V2X-R 必须已经包含固定 GT 和数据/划分/天气 manifest。

#### 阶段 1A：可选的执行配置测量

先比较 workers4/8；以下命令每份候选运行10次 warmup和40次测量更新，属于有界真实数据 benchmark，输出在独立的 `performance/`，不计入正式先导。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PERF="$RUNS/performance"
mkdir -p "$PERF"
BASE="$(python scripts/supplement.py configure --dataset v2xr --method deer_sbr \
  --variant AB --seed 17 --epochs "$EPOCHS" --data-root "$DATA" \
  --output-root "$PERF/base" --device "$DEVICE")"
check_config "$BASE"

# 仅生成两份 workers 候选和对应 benchmark 命令，不运行。
python scripts/supplement.py performance-configs --config "$BASE" --stage workers \
  --output-root "$PERF/workers_r1" > "$PERF/workers_r1.json"

# 最小测量：相同 seed/样本顺序，各 10 warmup + 40 measured updates。
while IFS= read -r C; do
  python project_tools/run_experiments.py --config "$C" --mode benchmark \
    --warmup 10 --steps 40 --timing-mode throughput
done < <(python -c 'import json,sys; print("\n".join(t["config"] for t in json.load(open(sys.argv[1]))["tasks"]))' \
  "$PERF/workers_r1.json")
```

随后按需逐项比较 prefetch1/2、cache0.5/1.0 GiB、cpu_threads2/4、activation checkpoint开/关、保存间隔100/500/1000。生成命令、RAM总量及选择标准见[性能测量步骤](docs/REPRODUCING.md#12-分阶段候选配置与有界服务器测量)。如果采用实测选出的执行参数，在**首次生成正式先导配置之前**设置下面的数组；不做测量则保持 `EXECUTION_ARGS=()`。

```bash
# WINNER 设为实测选择的完整候选 YAML 路径。
: "${WINNER:?set WINNER to the measured winning YAML}"
EXECUTION_ARGS=(--execution-config "$WINNER")
```

执行参数在先导与重复中保持一致。这个选项只复制资源、activation checkpoint和online保存间隔，不替换原生方法的优化器、batch、schedule或BN政策。

#### 阶段 2：S0 / AB / N2 的 seed17，随后 SUP

```bash
# EXECUTION_ARGS 默认空；若已冻结执行候选，则保持该数组用于后续同配方重复。
python scripts/supplement.py plan --suite pilot --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" --write-configs \
  | tee "$PLANS/pilot.json"
check_plan "$PLANS/pilot.json"
python scripts/supplement.py run-suite --suite pilot --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
  --execute --skip-completed | tee -a "$LOGS/pilot_e${EPOCHS}.log"

python scripts/supplement.py plan --suite pilot_sup --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" --write-configs \
  | tee "$PLANS/pilot_sup.json"
check_plan "$PLANS/pilot_sup.json"
python scripts/supplement.py run-suite --suite pilot_sup --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
  --execute --skip-completed | tee -a "$LOGS/pilot_sup_e${EPOCHS}.log"

PILOT_RUNS=()
for V in S0 AB N2 SUP; do
  PILOT_RUNS+=("$RUNS/supp_v2xr_deer_sbr_${V}_e${EPOCHS}_s17/train")
done
python scripts/supplement.py convergence --run-dirs "${PILOT_RUNS[@]}" \
  --output "$REPORTS/pilot_e${EPOCHS}_convergence.json"
```

每组输出为 `$RUNS/<run_id>/train/`，包含 `steps.rank0.jsonl`、`telemetry.rank0.jsonl`、逐轮 `validation_epochNNN.json`、online/selected和预算快照。`convergence` 输出JSON/CSV/PNG开发曲线。

#### 阶段 2A：需要时比较独立预算，再冻结 EPOCHS

以下示例只比较一个独立30轮配方；需要20轮时设 `E=20`，只有60轮仍持续改善时再考虑 `E=90`。不默认启动 `budgets` 的四组×四预算完整矩阵。

```bash
E=30
for SUITE in pilot pilot_sup; do
  python scripts/supplement.py plan --suite "$SUITE" --epochs "$E" \
    --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" --write-configs \
    | tee "$PLANS/${SUITE}_e${E}.json"
  check_plan "$PLANS/${SUITE}_e${E}.json"
  python scripts/supplement.py run-suite --suite "$SUITE" --epochs "$E" \
    --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
    --execute --skip-completed | tee -a "$LOGS/${SUITE}_e${E}.log"
done
BUDGET_RUNS=()
for V in S0 AB N2 SUP; do
  BUDGET_RUNS+=("$RUNS/supp_v2xr_deer_sbr_${V}_e${E}_s17/train")
done
python scripts/supplement.py convergence --run-dirs "${BUDGET_RUNS[@]}" \
  --output "$REPORTS/budget_e${E}_convergence.json"
```

依据开发曲线决定最终预算：两次降学习率均发生、最后10轮最佳开发均值改善低于0.3个百分点且无持续上升趋势，可作为平台候选。60轮配方的第30轮快照不是独立30轮终点。后续 `EPOCHS` 必须填已经冻结的20/30/60/90之一，不能由测试成绩决定。

#### 阶段 3：外部方法 seed17 先导

AttFuse、MDD、L4DR各30轮，CoBEVT为40轮；本阶段**不传 `--epochs`**。可分配另一张GPU执行，保留相同数据/固定GT协议及各方法原生配方。

```bash
python scripts/supplement.py plan --suite external_pilot \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" --write-configs \
  | tee "$PLANS/external_pilot.json"
check_plan "$PLANS/external_pilot.json"
python scripts/supplement.py run-suite --suite external_pilot \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
  --execute --skip-completed | tee -a "$LOGS/external_pilot.log"
```

#### 阶段 4：内部六组与外部四方法的三种子重复

内部 S0/A/B/AB/SUP/N2 × 17/29/43，共18项；外部四方法×三种子，共12项。完整同配方先导会跳过，历史20轮结果不自动计入新的长预算。下面 `EPOCHS=60` 应按上一阶段开发决定修改。

```bash
# 如开发证据决定使用 30 或 90，先修改此值，再生成全部重复配置。
EPOCHS=60
python scripts/supplement.py plan --suite internal_main --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" --write-configs \
  | tee "$PLANS/internal_main.json"
check_plan "$PLANS/internal_main.json"
python scripts/supplement.py run-suite --suite internal_main --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
  --execute --skip-completed | tee -a "$LOGS/internal_main_e${EPOCHS}.log"

python scripts/supplement.py plan --suite external_main \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" --write-configs \
  | tee "$PLANS/external_main.json"
check_plan "$PLANS/external_main.json"
python scripts/supplement.py run-suite --suite external_main \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
  --execute --skip-completed | tee -a "$LOGS/external_main.log"
```

#### 阶段 5：只用三种子开发证据冻结 F

需要 S0/AB/N2 × 17/29/43 的完整开发日志和selected。按Normal/Fog开发均值，在相对S0的三种子平均Normal降幅不超过1个百分点条件下选择AB或N2；相同均值选AB。F是共用结构，V2X-R复用该结构已经训练好的各seed权重。

```bash
DEV_RUNS=()
for V in S0 AB N2; do
  for S in 17 29 43; do
    DEV_RUNS+=("$RUNS/supp_v2xr_deer_sbr_${V}_e${EPOCHS}_s${S}/train")
  done
done
python scripts/supplement.py convergence --run-dirs "${DEV_RUNS[@]}" \
  --output "$REPORTS/main_development_e${EPOCHS}.json"
python scripts/supplement.py freeze-final --run-dirs "${DEV_RUNS[@]}" \
  --normal-tolerance-pp 1 --output "$FINAL_MODEL"
F_VARIANT="$(python -c \
  'import sys; from opencood.project.supplement import resolve_final; print(resolve_final(sys.argv[1]))' \
  "$FINAL_MODEL")"
printf 'Frozen F = %s\n' "$F_VARIANT"
```

#### 阶段 6：优先完成 M1 / M3 / M4

先seed17，再补29/43；三组均以AB为基础，F若为N2，不把这些控制称作N2的完整消融。需要在正文保留机制结论时补齐重复。

```bash
for SUITE in mechanism_pilot mechanism_repeat; do
  python scripts/supplement.py plan --suite "$SUITE" --epochs "$EPOCHS" \
    --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" --write-configs \
    | tee "$PLANS/${SUITE}.json"
  check_plan "$PLANS/${SUITE}.json"
  python scripts/supplement.py run-suite --suite "$SUITE" --epochs "$EPOCHS" \
    --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
    --execute --skip-completed | tee -a "$LOGS/${SUITE}.log"
done
```

#### 阶段 6A：按研究问题执行扩展机制

可选 M2/M5/M6/N1/NO_NULL × 17/29/43，共15项；按预先研究问题启用，不按测试表现保留或删选。

```bash
python scripts/supplement.py plan --suite mechanism_optional --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" --write-configs \
  | tee "$PLANS/mechanism_optional.json"
check_plan "$PLANS/mechanism_optional.json"
python scripts/supplement.py run-suite --suite mechanism_optional --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
  --execute --skip-completed | tee -a "$LOGS/mechanism_optional.log"
```

#### 阶段 7：rho / 支撑半径与第二协同骨干

此时必须已产生真实的 `$FINAL_MODEL`。rho和radius各3项，默认seed17且分别比较；Where2comm底座/+F各三seed，共6项，保留原生40轮。

```bash
for SUITE in rho radius; do
  python scripts/supplement.py plan --suite "$SUITE" --epochs "$EPOCHS" \
    --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
    --final-model-file "$FINAL_MODEL" --write-configs | tee "$PLANS/${SUITE}.json"
  check_plan "$PLANS/${SUITE}.json"
  python scripts/supplement.py run-suite --suite "$SUITE" --epochs "$EPOCHS" \
    --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
    --final-model-file "$FINAL_MODEL" --execute --skip-completed \
    | tee -a "$LOGS/${SUITE}.log"
done

# Where2comm 底座及 +F × 三种子；共 6 项，保留原生 40 轮，不传 --epochs。
python scripts/supplement.py plan --suite second_backbone \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
  --final-model-file "$FINAL_MODEL" --write-configs | tee "$PLANS/second_backbone.json"
check_plan "$PLANS/second_backbone.json"
python scripts/supplement.py run-suite --suite second_backbone \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
  --final-model-file "$FINAL_MODEL" --execute --skip-completed \
  | tee -a "$LOGS/second_backbone.log"
```

#### 阶段 8：V2X-Radar-C 真实协同扩展

仅在服务器已有官方协同数据后运行。先从官方train按sequence留出开发集，官方validate保留最终评测身份；输入是真实车辆ego及路侧。`CLEAR_SEQS` 只能填写已经核验的晴天train sequence；空数组会使需要类型/MDD参考监督的整组预检阻塞。

```bash
REAL=/data/v2x-radar-c
REAL_META=/data/metadata/v2xradar_supp_v1_20261024
REAL_RUNS=/data/runs/v2xradar_supp_v1_20261024
mkdir -p "$REAL_RUNS/plans"
# 仅填已经核验为晴天的官方 train sequence；不得按目录名称猜测。
CLEAR_SEQS=()
python scripts/supplement.py prepare --dataset v2xradar --data-root "$REAL" \
  --manifest-dir "$REAL_META" --dev-count 4 --split-seed 17 \
  --clean-sequences "${CLEAR_SEQS[@]}"

python scripts/supplement.py plan --suite real_cooperative --data-root "$REAL" \
  --manifest-dir "$REAL_META" --output-root "$REAL_RUNS" --device "$DEVICE" \
  --final-model-file "$FINAL_MODEL" --write-configs \
  | tee "$REAL_RUNS/plans/real_cooperative.json"
check_plan "$REAL_RUNS/plans/real_cooperative.json"
python scripts/supplement.py run-suite --suite real_cooperative --data-root "$REAL" \
  --manifest-dir "$REAL_META" --output-root "$REAL_RUNS" --device "$DEVICE" \
  --final-model-file "$FINAL_MODEL" --execute --skip-completed \
  | tee -a "$LOGS/real_cooperative.log"
```

主矩阵为S0/F/MDD-CORR/CoBEVT × 三seed。下面是独立L4DR候选槽，需要时执行；MDD没有合法clean参考时，先处理数据/监督条件，不用自然雨天冒充clean target。

```bash
# 独立 L4DR 槽；三 seed，不依赖 MDD clean target。
python scripts/supplement.py plan --suite real_cooperative_l4dr --data-root "$REAL" \
  --manifest-dir "$REAL_META" --output-root "$REAL_RUNS" --device "$DEVICE" --write-configs \
  | tee "$REAL_RUNS/plans/real_cooperative_l4dr.json"
check_plan "$REAL_RUNS/plans/real_cooperative_l4dr.json"
python scripts/supplement.py run-suite --suite real_cooperative_l4dr --data-root "$REAL" \
  --manifest-dir "$REAL_META" --output-root "$REAL_RUNS" --device "$DEVICE" \
  --execute --skip-completed | tee -a "$LOGS/real_cooperative_l4dr.log"
```

#### 阶段 9：K-Radar 真实天气扩展

先填写已有OS2-64、官方sparse Radar、v2.0 labels和官方split路径。主矩阵为S0/F/MDD-CORR/L4DR × 三seed；L4DR实际采用35轮AdamW/cosine、batch3，其余保留注册配方，不向原生suite传 `--epochs`。

```bash
KR=/data/kradar
KR_RADAR=/data/kradar_sparse_product
KR_LABEL=/data/KRadar_refined_label_by_UWIPL
KR_META=/data/metadata/kradar_supp_v1_20261024
KR_RUNS=/data/runs/kradar_supp_v1_20261024
mkdir -p "$KR_RUNS/plans"
python scripts/supplement.py prepare --dataset kradar --data-root "$KR" \
  --radar-root "$KR_RADAR" --label-root "$KR_LABEL" --manifest-dir "$KR_META" \
  --train-split /data/kradar_official_split/train.txt \
  --test-split /data/kradar_official_split/test.txt --dev-count 4 --split-seed 17
python scripts/supplement.py plan --suite real_weather --data-root "$KR" \
  --radar-root "$KR_RADAR" --label-root "$KR_LABEL" --manifest-dir "$KR_META" \
  --output-root "$KR_RUNS" --device "$DEVICE" \
  --final-model-file "$FINAL_MODEL" --write-configs | tee "$KR_RUNS/plans/real_weather.json"
check_plan "$KR_RUNS/plans/real_weather.json"
python scripts/supplement.py run-suite --suite real_weather --data-root "$KR" \
  --radar-root "$KR_RADAR" --label-root "$KR_LABEL" --manifest-dir "$KR_META" \
  --output-root "$KR_RUNS" --device "$DEVICE" --final-model-file "$FINAL_MODEL" \
  --execute --skip-completed | tee -a "$LOGS/real_weather.log"
```

K-Radar是单车双类别输入；MDD需要官方天气记录支持的晴天train参考。两个真实数据集分别使用自己的manifest与输出目录，均使用同一冻结F。

#### 任意阶段：从 online checkpoint 恢复

下面是内部AB/seed17的例子；其他任务使用其计划JSON中的准确run_id和配置路径。

```bash
C="$RUNS/configs/supp_v2xr_deer_sbr_AB_e${EPOCHS}_s17.yaml"
python scripts/supplement.py train --config "$C" --execute \
  --resume "$RUNS/supp_v2xr_deer_sbr_AB_e${EPOCHS}_s17/train/latest.rank0.pt" \
  | tee -a "$LOGS/resume_AB_e${EPOCHS}_s17.log"
```

`latest.rank0.pt` 保存online BN、optimizer、scheduler、sampler和RNG，用于resume；`selected.pt`用于开发选中权重评测；`budget_epochNNN.pt`是显式预算快照。`--skip-completed`只跳过已完成任务，不自动恢复中断任务。正式任务开始后保持源码、配置和数据身份一致。

多GPU优先分配互不重复的单卡任务。不同终端设置各自的 `CUDA_VISIBLE_DEVICES=<物理卡>`，进程内仍用 `device=cuda:0`；不要并发启动覆盖相同run_id的两个suite。workers、CPU线程、RAM缓存和磁盘预留按所有并发任务相加，容量示例见[总资源安排](docs/REPRODUCING.md#13-多-gpu-总资源安排)。

完成开发冻结和相应训练后，再执行[开发诊断与效率](docs/REPRODUCING.md#43-固定权重的开发诊断干预与效率)、[最终评测、鲁棒性与汇总](docs/REPRODUCING.md#6-最终评测鲁棒性与统计汇总)中的命令。训练期间每轮已经自动完成正式开发验证；最终测试不参与预算或结构选择。全部步骤的详细协议与阻塞条件仍维护在[复现指南](docs/REPRODUCING.md#补充实验复现入口schema-5)。

## Reproducibility Checks

The original release validation passed **159 regression tests**. A complete AB/Snow evaluation reproduced all six reported AP3D/APBEV metrics at IoU 0.3/0.5/0.7, and its prediction archive was byte-identical to the original experiment. Verification records are available in [`repro/pretrained_release_verification.json`](repro/pretrained_release_verification.json).

The supplemental implementation checks on **2026-10-23** covered 18 relevant test modules: **160 passed, 3 skipped, 0 failures or errors**. The three skipped checks require the complete prepared V2X-R dataset. In addition, **21 synthetic CPU model paths** passed forward/backward/inference checks, and **27 CLI calls** passed. Machine-readable records are available in [`results/supplement/`](results/supplement/), with scope and environment details in [VALIDATION.md](docs/VALIDATION.md).

Supplemental real-data training, final evaluation, and CUDA numerical/performance validation remain unexecuted. The local native-data preflight reports missing spconv/cumm, Open3D, pypcd, and compiled extensions. Synthetic checks establish functionality and do not supply experimental AP or efficiency claims.

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
paper/               Anonymous manuscript, LaTeX source, editable figure, supporting documents
opencood/            Models, datasets, training protocol, and evaluation
project_tools/       Training, prediction export, and diagnostic utilities
scripts/             Installation, data preparation, and reproduction entrypoints
cscwd_configs/       Project configurations, including the supplemental matrix and templates
repro/               Executed configurations, manifests, and verification records
results/reference/   Reference metrics and experiment identities
results/supplement/  Local supplemental implementation-check records
tests/               Module, geometry, protocol, and evaluation regression tests
docs/                Detailed reproduction and validation documentation
CITATION.cff         Machine-readable software and preferred paper citation
```

Raw datasets, local environments, generated predictions, and compiled extensions are excluded from Git. Pretrained checkpoints and the original evaluation metadata are distributed through GitHub Releases.

## Citation

If you use this implementation or its experimental artifacts, please cite the accompanying manuscript:

```bibtex
@unpublished{anonymous2026evidence,
  author = {{Anonymous Authors}},
  title  = {Evidence Routing and Supported Residuals for Cooperative {LiDAR}--{Radar} Perception},
  year   = {2026},
  note   = {Anonymous manuscript; authorship to be confirmed},
  url    = {https://github.com/Hqa77880011/CSCWD-DEER-SBR/blob/main/paper/paper.pdf}
}
```

The citation is also available in [`CITATION.bib`](CITATION.bib) and [`CITATION.cff`](CITATION.cff). The CFF metadata identifies the manuscript as the preferred citation for this repository.

## Acknowledgments and License

This implementation builds on [V2X-R](https://github.com/ylwhxht/V2X-R) and its OpenCOOD-based framework. We acknowledge the upstream authors and the related codebases credited in the retained [upstream V2X-R README](docs/V2XR_UPSTREAM_README.md), which provides the original benchmark documentation.

Please consult [`NOTICE.md`](NOTICE.md) and the applicable source-file headers for third-party attribution and license terms.
