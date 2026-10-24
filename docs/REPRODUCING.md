# 复现指南

原七组 20 轮结果使用下文原复现入口；新增实验使用本文末尾的 schema 5 入口。当前补充实验方案以 [13_补充实验方案与实施说明.md](13_补充实验方案与实施说明.md) 为准。新代码准备完成不代表实验完成。

## 固定的实验身份

实际实验为 V2X-R、`teacher_free_v4` / schema 4、`project_fix_v1` 评分协议、W 训练线、seed 17、20 epoch。七组为 S0/A/B/AB/SUP/N1/N2，各 20200 次有效优化更新；源数据 revision 为 `4c9ff1773b49517464714368d30778b5252bdbd0`。原始配置保存在 `repro/configs/`，训练和测试身份保存在 `results/reference/`。

原批次中 S0 从已核验的第 1000 步 online checkpoint 续训，其累计预算为 20 epoch。这里从同一种子重新初始化并完整训练，不依赖该服务器 checkpoint。CUDA/GPU/编译器差异可能造成数值差异；不承诺跨硬件逐位相同或 AP 精确到最后一位。

## 环境

从仓库根目录执行 README 中的 Conda 和安装命令。`requirements-repro.txt` 固定主要依赖，`requirements-repro-lock.txt` 记录本次服务器环境中这些依赖的传递依赖版本。安装脚本默认使用该文件作为 constraints 固定传递依赖。上游 `requirements.txt` 与 `INSTALL.md` 是历史配方。

PyTorch 为 2.1.2+cu121、torchvision 为 0.16.2+cu121、NumPy 为 1.26.3，spconv/cuMM 为 2.3.6/0.4.11。spconv wheel 名称中的 cu120 是已有环境使用的发布包名称。完整 CUDA toolkit 需提供 `nvcc`；驱动由宿主机提供。`project_tools/build_extensions.sh` 自动读取可见 GPU 架构。交叉编译时显式设置，例如 `TORCH_CUDA_ARCH_LIST=7.5`。

`bash scripts/install.sh` 构建 Cython 与 CUDA 扩展，执行 `pip check`，并运行合成 GPU 前向/反向、IoU/NMS 和 checkpoint 检查。详细结果写入 `outputs/environment_check.json`。环境安装不会启动正式训练。

<a id="minimal-server-validation"></a>

## 拉取最新代码后的最小验证（不训练）

适用于项目环境、CUDA扩展和数据已经准备好的Linux服务器。在已激活的项目环境中修改下面的仓库/数据路径；这些命令不安装、下载、重建manifest或启动训练。只执行小型算子检查和两个train样本的读取/collate，不创建优化器、更新BN或保存checkpoint。

```bash
set -euo pipefail
cd /path/to/CSCWD-DEER-SBR
git pull --ff-only origin main

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
CHECKS="outputs/verification/minimal_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CHECKS"

# L4DR实际batch CUDA算子与spconv；零优化更新，无checkpoint/数据集读取。
python project_tools/check_environment.py --checks l4dr spconv \
  --output "$CHECKS/cuda.json"

# V2X-R真实数据：新建检查用配置，不进入runner，不使用旧checkpoint。
V2XR_DATA=/path/to/datasets/v2xr
C="$(python scripts/supplement.py configure --dataset v2xr --method l4dr --variant S0 \
  --data-root "$V2XR_DATA" --manifest-dir "$V2XR_DATA" \
  --device cuda:0 --output-root "$CHECKS")"
python scripts/supplement.py check-data --config "$C" --samples 2 \
  | tee "$CHECKS/data_v2xr.json"
```

需要覆盖V2X-Radar-C/K-Radar时，在同一Bash会话中仅追加对应数据集的读取；无需重复GPU检查。两者要求新manifest已登记`sensor_sha256`；旧manifest缺证据会按设计拒绝，不自动补写或覆盖。

```bash
V2XRADAR_DATA=/path/to/v2xradar
V2XRADAR_META=/path/to/v2xradar_metadata
C="$(python scripts/supplement.py configure --dataset v2xradar --method l4dr --variant S0 \
  --data-root "$V2XRADAR_DATA" --manifest-dir "$V2XRADAR_META" \
  --device cuda:0 --output-root "$CHECKS")"
python scripts/supplement.py check-data --config "$C" --samples 2 \
  | tee "$CHECKS/data_v2xradar.json"

KR_DATA=/path/to/kradar
KR_RADAR=/path/to/official_sparse_radar
KR_LABEL=/path/to/revised_v2_0_labels
KR_META=/path/to/kradar_metadata
C="$(python scripts/supplement.py configure --dataset kradar --method l4dr --variant S0 \
  --data-root "$KR_DATA" --radar-root "$KR_RADAR" --label-root "$KR_LABEL" \
  --manifest-dir "$KR_META" --device cuda:0 --output-root "$CHECKS")"
python scripts/supplement.py check-data --config "$C" --samples 2 \
  | tee "$CHECKS/data_kradar.json"
```

通过条件：所有命令退出码为0；`cuda.json.status=passed`，只含`l4dr_cuda`和`spconv_gpu`两项、`baseline_optimizer_check_requested=false`；每个数据JSON给出两个实际train样本、agent/点数/来源与正anchor诊断，且`training_started=false`。空GT/零正anchor样本可合法出现，不据此删除样本。`check-data`只构建模型，不执行forward或loss；V2X-R实际走spconv预处理，两套真实补充数据走其登记的NumpyVoxelizer。

L4DR检查使用无最近邻/FPS最大值并列的4个点：FPS、ball-query（含严格半径边界及首邻点补齐）、three_nn索引与距离用解析已知答案严格比较，`rtol=atol=0`；FAD使用的PyTorch gather/加法epsilon权重/归约及feature梯度比较CPU/CUDA，`rtol=2e-6, atol=1e-7`只覆盖FP32设备运算顺序差异。spconv检查实际项目preprocessor的voxel内容/坐标/计数/collate已知答案，以及CUDA稀疏卷积输出与输入/权重梯度有限且非零。报告保留加载的PointNet++二进制路径、摘要与运行版本；本检查不证明完整FAD、真实字段物理单位、训练轨迹或AP等价，也不测速度。任一步失败即停止；本轮仅在本地检查了入口与CPU已知答案，服务器结果仍待执行。

这里必须保留`--checks l4dr spconv`；省略`--checks`仍使用原完整环境检查，会执行一次合成基线optimizer更新和checkpoint写读。不要把`run_experiments.py --mode smoke`加入本节：它会读取真实train并进行一次更新。

## 数据

下载源为 [hx24/V2X-R 的固定 revision](https://huggingface.co/datasets/hx24/V2X-R/tree/4c9ff1773b49517464714368d30778b5252bdbd0)，远端文件清单和 LFS SHA256 已保存在 `repro/metadata/remote_files.json`。真实可用的官方划分为 train=8083、validate=829、test=3166；不补造文献中多出的一个训练样本。

```bash
# 可先验证网络访问和固定源文件校验，仅读取很小的元数据文件
python scripts/prepare_data.py --data-root /disk/v2xr --probe-only
# 下载、校验、流式解包、配对检查，再构建固定 GT
python scripts/prepare_data.py --data-root /disk/v2xr --workers 2 --gt-workers 8
```

断网后重复同一命令可利用断点和已验证归档继续。必要时可通过 `HF_ENDPOINT` 指定兼容的 HTTPS 镜像，固定 revision 和哈希仍保持不变。数据准备使用 `aria2c` 和 `7z`，不会调用 Hugging Face SDK。每个压缩包在校验源 SHA256、成员 CRC32、记录保留文件哈希之后删除，降低磁盘占用。PNG 像素流被丢弃，保留校验过的尺寸元数据；模型不读取 RGB。

准备状态在数据目录同级的 `.v2xr_preparation/`，归档回执在数据目录的 `manifests/archives/`。已经完成下载和归档校验、只需重建元数据时使用 `--manifests-only`。工具会检查完整归档集合、模态配对、划分计数及重建 GT 的计数和语义哈希是否符合已有实验记录。

最终目录：

```text
v2xr/
  train/<sequence>/<agent>/<timestamp>{.pcd,_radar.pcd,_fog_0.060.pcd,_snow_2.5_2.0.pcd,.yaml}
  validate/...
  test/...
  data_manifest.json
  split_manifest.json
  weather_manifest.json
  image_sizes.json
  protocol_manifest.json
  eval_gt_manifest.json
```

已有本项目完整数据可以直接复用这些 manifest。只有裸点云而没有原始标注、图像尺寸元数据或哈希时，应重新运行数据准备，不能用任意图像尺寸替代。

## 配置、审计和训练

```bash
python scripts/reproduce.py configure --data-root /disk/v2xr --output-root /disk/runs/seed17
python scripts/reproduce.py audit --output-root /disk/runs/seed17
python scripts/reproduce.py preflight --variant all --output-root /disk/runs/seed17
python scripts/reproduce.py smoke --variant AB --output-root /disk/runs/seed17
```

`configure` 以实际执行的七组配置为模板，只重定位数据、输出和审计文件路径，保留科学参数及单 GPU 资源配置。配置已生成后，只需继续传同一个 `--output-root`；改变数据路径需要重新配置到新的输出目录。它不会覆盖内容不同的已有配置。

`audit` 在 32 个固定 train 样本上重放来源可追踪的删点/伪回波注入，证明辅助标签来自注册的受控扰动。Fog 的未知类型标签保持屏蔽。所有方法共享相同数据和扰动审计。训练入口继续核验审计绑定的源码、数据、种子和扰动参数，不能跳过该步骤或套用其他源码版本的报告。

`preflight` 仅读取配置并列出路径和正式训练阻塞项，本身不会训练；开始训练前检查路径均存在、`formal_blockers` 为空。`smoke` 只在 train 上执行一次优化更新，不提供验证 AP。

```bash
# 顺序训练七组
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py train --variant all --output-root /disk/runs/seed17
# 也可将不同组分配到独立 GPU 和终端，同一组保持单卡
CUDA_VISIBLE_DEVICES=1 python scripts/reproduce.py train --variant A --output-root /disk/runs/seed17
# 中断后，使用该组 online checkpoint 续训
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py train --variant AB \
  --output-root /disk/runs/seed17 \
  --resume /disk/runs/seed17/v2xr_tf4_W_AB_s17/train/latest.rank0.pt
```

每组输出在 `<output-root>/<run_id>/train/`。`latest.rank0.pt` 保存 online BN、优化器、调度器、采样器和 RNG 状态用于续训；`selected.pt` 保存验证选中的权重及重估后的 BN，用于评测。二者不可互换。复现过程中保持源码、配置、数据和审计文件稳定，否则身份校验会拒绝旧 checkpoint。

训练配置为 FP32、microbatch 4、累积 2、有效 batch 8，Adam lr=0.002、weight decay=1e-4、eps=1e-10，第 10/15 epoch 衰减至原来的 0.1 倍。所有 W 变体共享 Normal/Fog 输入计划、几何增强和 `provenance_v1` 扰动。不使用 teacher，也不使用 clean feature target。

每个 epoch 验证前，只在固定 32 个训练样本上进行 8 次无梯度前向重估 BN 统计，不增加优化步数。Normal/Fog 共用完整 `eval()` 和同一份统计。按两天气验证 AP3D@0.7 均值选模，相差不超过 1e-6 时保留较早 checkpoint。Snow 不参与该过程。

## 测试和结果

### 使用 Release 权重直接评测

[v1.0.0 Release](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/tag/v1.0.0) 提供七份原始 `selected.pt`、`evaluation-metadata.tar.gz`、`pretrained-release.json` 和 `SHA256SUMS`。权重未裁剪、未重写，SHA256 与 `results/reference/manifest.json` 的最终测试选择记录一致。N1/N2 选中第 18 轮，其余五组第 20 轮。

```bash
gh auth login
python scripts/pretrained.py download --variant all
python scripts/pretrained.py verify --variant all
CUDA_VISIBLE_DEVICES=0 python scripts/pretrained.py evaluate \
  --variant all --weather all --data-root /disk/v2xr \
  --output-root /disk/eval/pretrained-v1 --allow-test-evaluation
```

仅需一组时把 `--variant all` 改为 `--variant AB` 等名称；天气可选 `normal`、`fog`、`snow`、`all`。原始点云和标注按 README 准备，不包含在 Release 中。私有仓库的下载要求 `gh` 已登录且账号有仓库权限。手动下载时，权重和 `evaluation-metadata.tar.gz` 必须位于同一个 `--weights-dir`；默认是仓库内 `checkpoints/v1.0.0/`，已排除 Git 跟踪。

直接评测入口首先以仓库中的固定 Release 清单校验所有输入，再验证 checkpoint 内嵌的原配置和原配置哈希、源码哈希、selected 角色和 epoch、BN 统计哈希及训练校准记录。只有数据根目录、metadata 路径、监督审计路径和输出目录允许重定位；科学参数、GT、样本集合、后处理和 BN 均保持原身份。工具将原始 metadata 链接到输出目录的独立 `data/` 视图，保持原文件字节与哈希；原始点云/标注从你的数据目录读取，并由数据集加载器核验哈希。不需要重训或重新生成辅助监督审计，也不会在测试集上更新 BN。

每项测试输出 `predictions.jsonl`、对应 metadata sidecar、`scores.json.gz` 和 `summary.json`。摘要中的 AP 使用 0–1 数值，`AP3D07_delta_pp` 为与参考表相差的百分点，`reference_AP3D07_matches_1e_6` 给出该指标是否在 1e-6 的 AP 容差内。该字段不替代对跨硬件数值差异的检查。每项仍必须覆盖全部 3166 个样本和 30296 个 GT。工具拒绝覆盖已有预测，失败后保留现场；重试可使用新的输出目录。默认使用原最终测试采用过的 8 个 CPU 后处理进程；`--postprocess-workers 1` 可切换为相同科学实现的串行导出器。已有完整 Normal/Fog 验证等价记录保存在 `repro/pretrained_parallel_equivalence.json`，并行调度不改变解码、NMS、样本顺序或 AP 定义。

### 评测自己从头训练的权重

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py evaluate --variant all \
  --output-root /disk/runs/seed17 --allow-test-evaluation
```

该命令分别导出每组 Normal/Fog/Snow 的全部测试样本，并调用同一个独立评测器。预测 JSONL、哈希 sidecar 和结果 JSON 位于 `<output-root>/final_test/<variant>/`。有同名输出时会停止，避免意外覆盖已冻结结果；失败后请先检查该目录，使用原始 `export_predictions.py` / `evaluate_predictions.py` 定位恢复。

固定 GT 使用原始标注与标定，ego 中心 ROI 为 x∈[0,140)、y∈[-40,40)，固定 camera0 几何 FOV；不按天气点数删 GT，不截断 GT。评分是全局排序的 VOC2010 全点插值 AP，置信度下限 0.001、BEV NMS IoU=0.15、每样本最多 500 个预测。三天气共用同一组选中权重。测试为 3166 样本、30296 GT，天气均值为 Fog/Snow AP3D@0.7 算术均值。

`python scripts/reproduce.py results` 仅复核已有结果表的均值和增益。原始数据和数十 GB 训练输出不上传，原始选中权重通过 Release 提供。`scripts/reproduce.py evaluate` 评测本次配置下从头训练得到的 checkpoint；`scripts/pretrained.py evaluate` 在验证原始身份后迁移路径，评测 Release 中的原权重。两个入口分别记录相应配置和权重身份。

## 验证

```bash
bash scripts/test.sh
# 若已准备全量数据，可额外启用真实 train 样本/固定 GT 回归
CSCWD_DATA_ROOT=/disk/v2xr bash scripts/test.sh
```

默认回归使用合成输入；依赖全量数据的测试在数据不可用时明确 skip。`ci/github-actions.yml.example` 提供 GitHub Actions 模板，覆盖静态编译、结果核算与可移植入口回归，不提供 GPU。该模板未激活；启用时应核实运行环境与当前仓库的workflow权限。当前机器上实际做过的检查见 [验证记录](VALIDATION.md)。

历史中文设计和分析文档用于追溯。含旧 30 epoch、旧版本或服务器绝对路径的内容，以本指南和 `repro/configs/` 的 20 epoch 实际配置为准。结果局限包括单种子、单数据集、内部对照及未经测量的端到端资源成本。

## 补充实验复现入口（schema 5）

以下是一套在**已安装项目环境、已准备数据的 Linux 实验服务器**上分阶段执行的 Bash 命令，顺序为环境预检 → 先导训练 → 多种子与冻结 F → 消融及开发检查 → 数据扩展 → 最终评测。仓库本次只整理和检查命令，未连接服务器或启动训练。先完整读取对应阶段，再在服务器终端执行；可选实验块按方案启用。

新增批次为 schema 5 / `supplement_v1`。原 `scripts/reproduce.py`、`scripts/pretrained.py`、七份配置、参考结果和 Release 权重身份保留；两个旧入口继续从 [`repro/archive/legacy_v4_runtime.zip`](../repro/archive/legacy_v4_runtime.zip) 运行原科学源码。新批次不需要下载原 selected 权重，不向旧实验输出目录写入。

### 0. 服务器变量、配置和输出约定

先按服务器实际挂载修改下面的路径。`REPO` 是当前 Git 仓库，`DATA` 是已准备的 V2X-R 根目录；`RUNS` 使用一个新的批次目录。所有后续命令在同一 Bash 会话、仓库根目录执行。

```bash
# 按服务器实际位置修改这三项；不要填本地 Windows 路径。
export REPO=/data/projects/CSCWD-DEER-SBR
export DATA=/data/v2xr
export RUNS=/data/runs/supplement_v1_20261023

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

`CUDA_VISIBLE_DEVICES=0` 选择服务器物理卡；如果改成 `2`，进程内仍使用 `DEVICE=cuda:0`。本配方每个任务单卡，`run-suite` 顺序执行。不要在不同终端同时写同一个 run_id。`EPOCHS=60` 是先导候选；仅依据开发曲线决定后续预算，正式重复前统一冻结为 20/30/60/90 之一。

| 内容 | 配置或输出路径 |
|---|---|
| 模板与矩阵（仓库内，只读） | `cscwd_configs/supplement/{v2xr,v2xradar,kradar}.yaml`、`matrix.json` |
| 展开的任务清单 | `$PLANS/<suite>.json`；包含每项 `config`、run_id、轮数、batch 和权重路径 |
| 每项完整配置 | `$RUNS/configs/<run_id>.yaml` |
| 内部组 run_id | `supp_v2xr_deer_sbr_<variant>_e<EPOCHS>_s<seed>` |
| 原生外部组 run_id | `supp_v2xr_<method>_S0_e<原生轮数>_s<seed>_native` |
| 训练产物 | `$RUNS/<run_id>/train/` |
| 开发冻结记录 | `$FINAL_MODEL`，即 `$RUNS/frozen_final.json` |
| 最终评分与统计 | `$RUNS/final_eval/selected_v1/`、`$REPORTS/` |

配置存在且内容相同可复用；内容不同时拒绝覆盖。更换数据、配方或科学源码应使用新批次。已有 JSON 评分、冻结记录和预测也拒绝覆盖，重测时换一个输出子目录，例如 `selected_v2`，不要删除原结果。

### 0.1 有效配置、参数范围与恢复身份

配置只按以下顺序生成一次：schema5 defaults → 数据集YAML → method/variant及原生候选 → 显式路径/预算/科学参数 → `--execution-config`的限定执行字段 → schema校验。CLI省略 `--epochs` 时采用模板预算：V2X-R为60，V2X-Radar-C/K-Radar为30；suite则使用矩阵中的预算，`--native-recipe`/矩阵native槽采用对应作者配置。独立20/30/60/90轮严格匹配[10,15]/[15,23]/[30,45]/[45,68]，不会仅改轮数仍沿用旧schedule。生成后不再静默改科学参数；推理/训练显式 `--device` 改动会进入本次配置身份。

| 字段 | 默认值、单位、适用范围 | 实际消费者/校验 |
|---|---|---|
| `model.activation_checkpoint` | 内部组true；Where2comm+局部模块true；无局部模块的外部方法false。S0没有可重算分支 | `PointPillarDeerSbr.fuse_encoded`，仅training且启用grad时重算无BN的DEER/SBR，不重算PFN/backbone、MDD参考分支或FAD |
| `model.efficient_features` | 内部/Where2comm+局部模块true；其他外部方法false | 路由/残差与 `ProjectLoss` 的等价特征/稀疏辅助loss路径；不控制原生外部网络。无局部模块时开启开关或设置内部loss权重会被拒绝 |
| `resources.num_workers` | 8个进程/**每loader/每任务**；0表示主进程读取 | `make_loader`；0时resolved config明确将persistent_workers归为false；不会启动worker |
| `prefetch_factor` / `persistent_workers` / `pin_memory` | 2个batch/worker、true、true | DataLoader参数；prefetch和persistent仅workers>0生效，pin_memory接到loader；`to_device`使用non_blocking，CPU不会因此变成异步CUDA传输 |
| `raw_cache_gib_per_worker` / `ram_budget_gib` | .5 GiB/worker/loader；64 GiB/**任务**，1GiB=2^30 bytes | `ByteLRU`缓存原始解析数组/标注；workers=0仍有每dataset进程内缓存。schema核算train+全部开发域+BN stream，缓存上限总和≤该任务RAM预算一半 |
| `cpu_threads` | 主进程PyTorch intra-op线程4；worker PyTorch固定1 | runner `torch.set_num_threads` / `_worker_init`。不是整任务CPU核上限，也不会替你限制NumPy/OMP/MKL/OpenBLAS；启动前设置上面的三个环境变量。preflight和startup记录环境值 |
| `checkpoint_interval_updates` | 每100个**有效optimizer更新**；内部相当于200个microstep | runner保存online状态；逐epoch验证/保存、selected和budget snapshot独立于该间隔，诊断末尾仍实际保存online checkpoint |
| `training.batchnorm` | 公共PFN/backbone固定eps=.001、momentum=.01；内部V2X-R固定train重估32样本、microbatch4；外部/真实数据running时refit_samples=0 | 不接受未接线的eps/momentum或未注册的refit采样/epoch/增强配置。L4DR FAD自身保留作者BN默认eps=1e-5、momentum=.1，单点使用已有running统计，不套公共BN参数 |
| `data.base_hypes` / `model.local_variant` | 内部为baseline_full_data_check；外部为方法对应的NATIVE_HYPES；local_variant仅Where2comm可用 | schema→`configure_hypes`→`SupplementalModel`；错误方法/YAML、变体/模块或comparison标签组合直接拒绝 |
| `loss.*` / legacy字段 | 内部三权重为.1/.001/.01；无局部模块外部三者为0，MDD/FAD原生额外目标独立保留 | 外部不消费DEER/SBR loss权重。schema5不消费legacy matching/audit：matching只允许保留默认占位、supervision_audit/amp_audit必须null；不伪装成已执行审计 |

其他局部模型参数仅在对应分支/variant存在时起作用，例如rho/attention/support属于SBR，g_min属于multiplicative类型先验；共同配置中的这些字段不表示原生AttFuse、CoBEVT或MDD执行了内部模块。训练预算的FP32/micro4/累积2/有效batch8适用于内部受控组；原生外部候选的batch见下表，执行配置不会覆盖它们。

| `_native`候选 | 轮数/有效batch | optimizer：lr / weight decay / eps | 实际schedule |
|---|---|---|---|
| V2X-R AttFuse | 30 / 8 | Adam：.002 / 1e-4 / 1e-10 | MultiStep 10/15，gamma=.1 |
| V2X-R MDD、L4DR | 30 / 2 | Adam：.002 / 1e-4 / 1e-10 | MultiStep 10/15，gamma=.1 |
| V2X-R CoBEVT | 40 / 2 | Adam：.001 / 1e-4 / 1e-10 | 项目epoch warmup10+cosine，warmup_lr=2e-4、min_lr=2e-5 |
| V2X-R Where2comm及+F | 40 / 2 | Adam：2e-4 / .01 / 1e-10 | 同一个项目epoch warmup10+cosine，warmup_lr=2e-5、min_lr=5e-6 |
| K-Radar L4DR | 35 / 3 | AdamW：.001 / .01 / 1e-8 | 作者 `TYPE_TOTAL_ITER=every`：每有效更新step，T_max=实际完整更新数/epoch，min_lr=.0001，跨轮不重置 |

真实数据上的MDD/CoBEVT等继承所列V2X-R候选的超参数来源并适配目标数据，不能称已取得各目标数据集的原论文配方。`cosine_warmup`的epoch实现与作者timm调用链存在时间单位差异；MDD的参考目标/beta和Where2comm完整论文机制也有明确区别。采用的论文版本、作者commit、A/B/C/D分类和输入/监督/成本边界统一见[方案§8.2](13_补充实验方案与实施说明.md#82-原文作者版本与当前适配的核查结论)，不能仅看 `_native` 后缀判断完成了原生复现。

`run_id`区分dataset/method/variant/budget/seed/native及单因素；它不是所有参数的编码。完整YAML与resolved config保存全部字段，已有不同内容配置拒绝覆盖；data/GT/base-hypes/frozen-F及Python源码身份由artifact metadata绑定。online resume还核对完整配置、角色和状态。`--skip-completed`要求成功报告、完整配置、当前源码/数据/初始化及selected内嵌身份/角色均匹配；缺文件或内容变化返回不匹配，随后实际启动会明确报错，不静默采用另一来源。新源码需新批次，不能放宽检查续接旧schema5 checkpoint。

`retrained`和`fine_tuned`评测必须等于checkpoint内嵌的训练transfer；metadata中的`training_transfer`与配置一致。微调通过 `configure --transfer fine_tuned --initialize-from <source-selected.pt>` 将绝对来源路径和`experiment.initialization_checkpoint_sha256`一并冻结到配置，run_id另加transfer和来源摘要。训练默认使用该配置路径；训练入口的`--initialize-from`仅可重复指定同一路径。首次启动、online恢复、完成判断均要求该文件存在且字节相同；替换来源须生成新配置/批次，不迁移优化器/RNG。独立推理使用checkpoint内嵌的来源摘要，无需保留初始化文件，但不能改成另一来源或重训标签。`zero_shot`继续按明确的架构匹配规则跨数据集适配，并保留实际训练transfer和来源记录，禁止目标训练/BN拟合；不自动转换单类/双类检测头。

共享初始化按dataset/method/seed/transfer/完整来源摘要隔离；同一来源下的S0/AB/N2等配对变体仍检查共同底座tensor身份。只改变variant不会建立新的底座，重训和不同微调来源则互不冲突。

`preflight.execution_resource_budget`报告每任务loader流数、worker/预取batch/原始缓存上界，需按同时运行任务相加；它不包含模型、manifest、pinned队列和系统页缓存字节，不代替进程树/cgroup峰值实测。

`source_tree_hash()`继续保持Python源码算法；schema5的L4DR另在artifact metadata绑定`l4dr_pointnet2`：仅实际使用的PointNet++ batch扩展C++/CUDA/header、构建脚本、wrapper所引用二进制文件内容，以及Python ABI/PyTorch/CUDA runtime/C++ ABI。`environment`还记录实际加载路径和当前构建环境变量；这些变量不声称是该二进制的历史编译命令。源码相同、二进制相同均不证明二者来自同一次构建或算子数值等价；`operator_numerics=not_verified_by_identity`明确保留此边界。本地无扩展时记录`binary.status=unavailable`，没有CUDA数值证据。新schema5的恢复、选中权重加载和完成判断要求相同扩展证据；缺证据的旧schema5 checkpoint拒绝复用。原七组配置、Release清单、legacy源码包及其身份算法保持原样。

下面两个终端函数用于逐配置预检和读取两个真实 train 样本。`preflight` 的退出码本身不代表没有阻塞，函数会另外检查 `formal_blockers` 和输入路径；检查通过后才进入训练。它们不执行优化更新。

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

### 1. 环境与数据预检

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

通过条件：`environment.json` 为 `ready`；`environment_gpu.json` 为 `passed`；每份先导预检的 `formal_blockers=[]`，train 样本可以真实读取且正 anchor/agent/点数合理。完整 GPU 检查含一次**合成**基线优化及 checkpoint 写读，不是数据集训练，也不报告 AP；它不能替代 L4DR FAD 等新路径的真实 CUDA 运行。缺少扩展时先按本文环境部分构建并重新检查，不通过导入成功推断数值正确。

V2X-R 应已有 `data_manifest.json`、`split_manifest.json`、`weather_manifest.json`、`image_sizes.json`、`protocol_manifest.json` 和 `eval_gt_manifest.json`。本套新增命令不调用数据下载脚本，也没有合成数据回退。

### 1.1 性能证据与计时口径

2026-10-23 本地检查中，`outputs/`、`results/`、`repro/` 没有正式训练的 `steps.rank*.jsonl`、`telemetry.rank*.jsonl` 或逐轮 `validation_epoch*` 原件。`results/reference/validation_scores.csv` 仅记录选中轮次和分数，不能恢复训练曲线、BN 或保存耗时。已有 [并行导出核验摘要](../repro/pretrained_parallel_equivalence.json) 保留以下历史量，均来自旧 AB selected、829 个验证样本、8 个后处理进程：

| 历史摘要（秒） | Normal | Fog | 可解释范围 |
|---|---:|---:|---|
| 记录的整项 wall time | 237.318 | 227.798 | 旧验证导出核验任务，非新增训练吞吐 |
| loader wait 累计 | 1.310 | 1.440 | 主进程等待，不是全部文件 IO/预处理工作量 |
| H2D / forward / decode | 9.447 / 104.412 / 27.416 | 6.701 / 94.721 / 31.038 | 原归档 `scripts/parallel_export.py` 对这些阶段前后同步 CUDA |
| postprocess 累计 | 505.769 | 580.624 | 多进程工作时长之和，可以大于 wall time，不能相加作耗时占比 |

这些记录提示前向与 CPU 后处理值得测量，但没有本轮前后对照；不能据此报告此次加速比。原始点云读/解析、训练几何与体素预处理、反向、optimizer、BN refit、AP scoring、checkpoint 保存均缺少可用的本轮实测。

新训练入口默认 `--timing-mode throughput`：保留每个 microstep 的 loss 有限性拦截、每次更新的合并梯度判定和更新结束的 CUDA 同步；取消其余阶段计时专用栅栏。`update_wall_seconds` 和 `effective_samples_per_second` 包含 loader 等待、CPU 元数据、传输、前后向、梯度检查、optimizer 完成及 loss 标量汇总，排除 checkpoint/validation。`samples_per_second_including_periodic_io` 另包含周期保存与 step 日志；最终保存单列在 telemetry，不能将它遗漏后声称整任务完成时间。

正常模式的 `h2d_seconds / compute_seconds / optimizer_seconds` 标为 `host_dispatch_and_required_waits`：异步阶段会重叠，不能当作 GPU kernel 耗时。`--timing-mode synchronized` 才提供逐阶段同步诊断，另开启数据 worker 计时；它会改变流水重叠，不用于速度排行。`worker_data_timing.raw_read_parse_seconds` 含读取、原有身份校验和解析，`preprocessing_seconds` 含几何、扰动、voxel、观测图和 anchor target；它们是样本工作时间累计，排除 collate/IPC，不能与主进程 loader wait 相加。正常验证遥测不再隐式启用同步 profiler。

验证的 `dataset_loader_setup`、`refit_complete.seconds`、`validation_weather_complete.stage_seconds.scoring`、`validation_complete.seconds` 和 `checkpoint_commit.{seconds,serialization_fsync_seconds,checkpoint_bytes}` 分别支持定位构建、BN 重估、AP 评分、整个验证区间和保存。整个验证区间还包含 selected/预算快照及记录写盘，不能把其中子项再重复累加。checkpoint 的 `seconds` 含 BN 检查与序列化/fsync，不含随后日志持久化；step 中的 `checkpoint_seconds` 包含这次 commit 的日志开销。

### 1.2 分阶段候选配置与有界服务器测量

本节命令**仅供已准备数据的服务器执行，本地未运行真实 benchmark**。候选定义在 [performance_candidates.json](../cscwd_configs/supplement/performance_candidates.json)，由同一 schema 5 校验器生成完整 YAML；不修改三份模板、原七组配置或训练配方。默认始终 FP32、microbatch 4、累积 2、有效 batch 8。

| 次序 / `--stage` | 单因素候选 | 固定其他设置及进入条件 |
|---|---|---|
| 1 / `workers` | 4、8 | 从同一 AB/seed17 配方开始；先定位 data wait |
| 2 / `prefetch` | 1、2 | 基于上一步胜出配置；比较队列峰值 RAM |
| 3 / `cache` | 每 worker 0.5、1.0 GiB | 只有 raw read/parse 占用值得优化且总 RAM 允许；需看第二次开发遍历命中率 |
| 4 / `cpu` | 主进程 `cpu_threads` 2、4 | OMP/MKL/OpenBLAS 均先设 1，避免每个 worker 再开 BLAS 线程组 |
| 5 / `activation` | true、false | 使用 AB 等确有局部模块的内部组；关闭后必须留出显存余量；S0/原生外部方法不生成无效比较 |
| 6 / `checkpoint` | 100、500、1000 updates | 根据保存代价和可接受的中断丢失窗口选取；逐轮保存/验证/选模不变 |
| 条件扩展 / `workers12` | 12 | 8 相对 4 有稳定收益，CPU/RAM/存储仍有余量时才测 |

已经生效的路径为：`make_loader` 的 pin memory/persistent workers/prefetch；`to_device(..., non_blocking=True)`；optimizer 的 `zero_grad(set_to_none=True)`；模型与 loss 的 `efficient_features=True`。`cpu_threads` 进入主进程 `torch.set_num_threads`，worker 的 PyTorch 线程固定为 1。activation checkpoint 只重算无 BN 的局部分支；外部方法保留原生计算和配方。`raw_cache_gib_per_worker` 现同时接入 V2X-R 和两套真实数据 raw reader。

runner 只复用 dataset/loader、manifest、固定预处理器及有界的原始点云/标注缓存；每轮验证前重新核对数据身份，V2X-R 文件哈希检查保留，文件 stat 或预期哈希变化不会命中旧项。V2X-Radar-C/K-Radar在manifest中登记全部实际读取的LiDAR/Radar内容摘要，K-Radar Radar相对`radar_root`解析；run/load/completion边界扫描登记文件并生成`sensor_inputs_sha256`，不在每个batch重扫整个数据集。自定义reader的cache miss对同一份将被解析的字节校验，cache hit也核验当前来源内容，包括大小/mtime不变的替换；不匹配直接拒绝。缓存因此省去解析，不省去来源字节检查。旧manifest缺少`sensor_sha256`时明确拒绝，须从已审阅输入准备新的manifest，不能补写摘要后继续旧实验。GT、pose、calibration、划分和天气字段已冻结在manifest内，由`gt_manifest_sha256`绑定；运行时不重读原annotation/标定/description文件来改写这些字段。

真实点云在校准/增强前复制，训练的样本键控天气、扰动、点序和标签每次重新计算。内容检查的IO/摘要时间（含cache hit）计入诊断的raw read/parse，不归入预处理计算。缓存不含模型特征、预测或随机增强后的训练结果；每轮仍用当前权重重新拟合 BN 和评分，退出时回收所有 loader worker。

训练、验证导出、机制诊断和独立效率路径现均按实际方法筛选模型输入。无局部DEER/SBR模块的外部方法不生成observation maps，验证阶段也不生成无人消费的常规辅助目标；可视化显式请求地图。外部训练的类型覆盖日志仍消费辅助目标，保留其生成和原语义；MDD reference、FAD监督和来源trace保留。L4DR仅传raw Radar给FAD，不再提前生成随后废弃的Radar voxel；V2X-R仍消耗原Radar点序permutation以保持局部RNG状态。导出与真实数据evaluator的两次NMS暂保留：导出后预测还供可视化/bootstrap/距离分组使用，独立evaluator仍校验并处理输入，未新增“已后处理”信任契约。这些改动没有吞吐或显存收益实测。

2026-10-24新增的执行优化只删除Where2comm无用的feature count/.item、在H2D前筛除外部网络/loss未消费的观测图和类型target，以及将BN快照读取合并为浮点buffer/int64计数两批；没有删除有限性拦截、改变多尺度重复前向或BN更新次数。L4DR的detach、插值epsilon与K-Radar optimizer/scheduler修复另计为方法实现更正，不能把它们的训练变化解释为等价提速。CPU已知答案与严格等价范围见[验证记录](VALIDATION.md)；没有本轮真实数据或CUDA提速比。

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

主输出为 `<候选 output-root>/<run_id>/benchmark/{report.rank0.json,steps.rank0.jsonl,telemetry.rank0.jsonl,latest.rank0.pt}`，清单 `argv` 就是有界 benchmark，`main_outputs` 指向该目录。该模式确实进行 train 样本优化更新，但最多 **200 updates（warmup+steps）**，不进入正式训练目录、不产生 selected，不计入正式先导。重新测量用 `workers_r2`、`workers_r3` 等新目录，交换候选执行顺序，避免把 OS 文件缓存预热差异误认为参数收益。

只在上述结果有需要时进行分阶段同步定位。下面生成独立诊断路径，`C` 从实际生成清单读取；共 2+8 次更新，然后对每个开发域前 32 样本执行两遍诊断，并在每遍前严格按原 32 个 train 样本重估 BN：

```bash
python scripts/supplement.py performance-configs --config "$BASE" --stage workers \
  --output-root "$PERF/diagnostic_r1" > "$PERF/diagnostic_r1.json"
C="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["tasks"][0]["config"])' \
  "$PERF/diagnostic_r1.json")"
python project_tools/run_experiments.py --config "$C" --mode benchmark \
  --warmup 2 --steps 8 --timing-mode synchronized --diagnose-validation-samples 32
```

两遍分别观察首次构建/读取和 loader/raw cache 复用，模型权重不在两遍之间更新。子集评分只服务管线计时，不写正式逐轮评分文件，不参加选模，也不能当作开发 AP 或外推全量缓存收益。正式逐轮验证仍覆盖全部 Normal/Fog，BN 规模、频率和选模规则不变。若中断，不继续额外诊断。

每阶段从**实际测量后选择的完整 YAML**继续。例如将 `WINNER` 设为 workers 胜出配置的真实路径后，运行下一阶段；不要同时展开全部维度：

```bash
# WINNER 必须先设为上一步实测选择的 YAML，不能把本示例当成已选出最优值。
: "${WINNER:?set WINNER to the measured winning YAML}"
python scripts/supplement.py performance-configs --config "$WINNER" --stage prefetch \
  --output-root "$PERF/prefetch_r1" > "$PERF/prefetch_r1.json"
# 按需将 stage 改为 cache / cpu / activation / checkpoint；每步重新选择 WINNER。
```

选择标准：同一设备、软件、初始权重、agent/点数分布与测量窗口，至少三次运行比较每次正常模式吞吐及 step p50/p95。收益须超过运行波动；可预先采用“中位吞吐提升至少 5%，且三次方向一致”作为执行配置接受条件，而非论文精度结论。没有稳定收益时选较低资源候选。检查无非有限数值、无新增 allocator OOM、无 swap，峰值 allocated 建议不超过单卡可用显存 85%；主机按整个任务进程树/cgroup 的峰值内存核算，单个进程 RSS 不能替代总 RAM。训练缓存不足一轮时，短连续采样不能证明跨 epoch 命中收益；以先导的完整第二轮再核验。

checkpoint 的 500/1000 间隔大于有界窗口，不能在 200 updates 内声称验证了其周期吞吐。所有 benchmark 都实际写最终 online checkpoint，并测量写盘；100 间隔可用 `--warmup 20 --steps 100` 在新目录实测一次周期保存。500/1000 先按实测单次保存 `C` 秒估算摊销 `C/500`、`C/1000`，明确为估计；最终在 seed17 先导中核验周期事件。每轮末保存及 selected/预算快照另计。间隔越大，意外退出时最多丢失的未提交更新越多；SIGINT/SIGTERM 会在完成有效更新后保存。

冻结执行参数后设置 `EXECUTION_ARGS=(--execution-config "$WINNER")`，在**首次生成正式先导配置前**使用下节命令。这个参数只复制资源字段、activation checkpoint 和 online 保存间隔，保留各方法原生 batch/optimizer/schedule、BN、损失和选择规则。执行配置也属于严格恢复身份；正式任务开始后不要原地改配置或通过放宽恢复检查套用新参数。AMP/BF16/TF32、microbatch 和 `torch.compile` 仅保留为另行登记并完成数值验证后的研究候选；本次没有提供绕过 FP32 schema 的入口。

### 1.3 多 GPU 总资源安排

多卡优先独立任务，不改变单任务有效 batch 或训练 BN。下面是**容量规划示例，不代表已知服务器硬件**：2 张足够显存的 GPU、32 个可分配 CPU 核、128 GiB RAM 时，可让 GPU0/GPU1 各一任务，每任务先用 workers=4、prefetch=1、cache=0.5、cpu_threads=4、RAM budget=56 GiB，OMP/MKL/OpenBLAS=1；两任务预算合计112 GiB，另留16 GiB给系统，CPU每任务预留12核、其余8核供系统与文件服务。训练与一个评估流可能短时并行预取，需为两组 worker 加主进程线程留余量。

V2X-R 内部组同时最多保留 train、BN refit、Normal、Fog 四组 loader；workers=4 时共 16 个 worker 进程/任务，多数在非所属阶段空闲，raw cache 上限 `4×4×0.5=8 GiB/任务`。workers=8、cache=1 时为 `4×8×1=32 GiB/任务`，两任务合计 64 GiB **仅缓存**，还需 manifest、预取张量、pinned memory、Python 进程和系统页缓存；每任务56 GiB预算时该候选会被拒绝，不能为了生成配置虚增RAM预算。schema 要求所有 loader 的 raw cache 总额不超过该任务 `ram_budget_gib` 的一半；这个门槛不替代整机实测。外部 running-BN、真实数据单开发域按实际流数计算，workers=0 仍有进程内缓存。

磁盘按实测 checkpoint 大小 `S` 估算每任务常驻 `(2 + retain_epochs数量)×S`（online、selected、预算快照），另预留至少一份 `S` 给原子替换临时文件，以及初始化、日志和至少 20% 空闲空间；多任务相加。原子保存、fsync 和角色元数据保留，未采用异步保存。不同 GPU 的验证/保存尽量错峰，共享盘繁忙时先降低任务并发，不用测试成绩删选实验。

执行顺序仍为 S0/AB/N2 的 seed17，随后 SUP；只有相同完整配方的先导计入 17/29/43 重复。开发曲线冻结预算后，先完成关键三种子与 M1/M3/M4，再按原研究问题开展 M2/M5/M6/NO_NULL 等扩展；F 只由 S0/AB/N2 三种子开发证据冻结。外部方法各自保留原生配方。把已有清单中互不重复的配置分给不同 GPU 即可，每个进程用 `CUDA_VISIBLE_DEVICES=<物理卡>`、配置 `device=cuda:0`；不要并发启动两个覆盖同一 suite/run_id 的队列。`run-suite --skip-completed` 继续要求成功报告、selected 存在且 resolved config 完全匹配。

### 2. 先导训练、独立预算与恢复

先导为 S0/AB/N2 × seed 17，加 SUP 单列。只有下列含 `--execute` 的命令才启动正式训练。

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

主输出包括 `steps.rank0.jsonl`（有效 batch、实际更新、lr、loss 等）、`validation_epochNNN.json`（完整 Normal/Fog 开发评分）、`state.json`（选中轮次）和各角色权重。收敛工具另写同名 CSV/PNG。方案的平台候选条件为：两次降学习率已发生，末 10 轮最佳开发均值改善低于 0.3 个百分点，且无持续上升趋势。只用开发曲线决定是否扩展 90 轮。

需要比较独立短配方时，执行下块并选择 `E=20` 或 `30`；只有 60 轮仍持续改善时才另设 `E=90`。这三种候选均使用独立 run_id，从头初始化，学习率节点分别为 [10,15]、[15,23]、[45,68]；60 轮为 [30,45]。

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

`budgets` suite 可一次展开四组 × 四预算，但不作为默认启动命令。60 轮中的 `budget_epoch030.pt` 是 `long_recipe_intermediate`，不等同于独立 30 轮终点。预算比较按配方分别读取开发日志和明确身份的预算快照。

外部方法先各跑 seed 17，保留原生候选轮数、batch、optimizer 和 schedule；**不传 `--epochs`**。AttFuse/MDD/L4DR 在 V2X-R 为 30 轮，CoBEVT 为 40 轮。L4DR 为含 FAD/MME/MSGF 的共同四通道/检测协议适配。

```bash
python scripts/supplement.py plan --suite external_pilot \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" --write-configs \
  | tee "$PLANS/external_pilot.json"
check_plan "$PLANS/external_pilot.json"
python scripts/supplement.py run-suite --suite external_pilot \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
  --execute --skip-completed | tee -a "$LOGS/external_pilot.log"
```

中断后仅用同配方的 online checkpoint 恢复，下面是 AB/seed 17 的完整例子。`--skip-completed` 只跳过配置完全一致且成功产出 selected 的任务，不自动恢复中断任务。

```bash
C="$RUNS/configs/supp_v2xr_deer_sbr_AB_e${EPOCHS}_s17.yaml"
python scripts/supplement.py train --config "$C" --execute \
  --resume "$RUNS/supp_v2xr_deer_sbr_AB_e${EPOCHS}_s17/train/latest.rank0.pt" \
  | tee -a "$LOGS/resume_AB_e${EPOCHS}_s17.log"
```

`latest.rank0.pt` 保存 online BN、优化器/调度器/采样器/RNG，仅供 resume；`selected.pt` 用于开发选中权重评测；`budget_epochNNN.pt` 用于显式预算快照评测。不能用 selected 续训，也不能用旧 20 轮 resume 冒充独立 60 轮配方。

新任务继续调用 `project_tools/run_experiments.py`；本套矩阵均为 `retrained`。另行微调须先通过configure绑定transfer、初始化文件和内容摘要，详见[配置与恢复身份](#01-有效配置参数范围与恢复身份)；底层`--initialize-from`不可覆盖配置来源，不迁移优化器/RNG。`zero_shot`配置禁止目标训练和目标BN拟合。当前没有将V2X-R单类权重自动转换为K-Radar双类权重的路径。

### 3. 多种子主矩阵与冻结唯一 F

先依据先导开发曲线设置 `EPOCHS`，然后固定预算运行内部六组 × 17/29/43，共 18 项。与先导完全相同的四项由 `--skip-completed` 复用；若更换预算，则是新的 18 项配方。外部四方法 × 三种子共 12 项，同样复用已成功完成的先导。

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

冻结需要匹配的 S0/AB/N2 × 三 seed 完整开发日志和 selected 权重；依据 Normal/Fog 开发 AP3D07 均值，在相对 S0 的三 seed 平均 Normal 降幅不超过事先登记的 1 个百分点条件下，选出一个 AB 或 N2，完全相同则 AB。若没有合格候选则停止并检查开发证据。不得按测试天气或 seed 选结构。V2X-R 的 F 直接引用该结构各 seed 的现有 selected，不重复训练一套名为 F 的权重。

### 4. 消融、第二骨干与开发诊断

#### 4.1 从头训练的机制消融

M1/M3/M4 为 seed 17 先导及 29/43 重复，共 9 项；M2/M5/M6/N1/NO_NULL 是按方案需要启用的扩展机制块，共 15 项。所有这些控制以 AB 为基础；F 若选 N2，不能把 AB 消融解释为 N2 的完整消融。

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

扩展机制块需要时执行；最终评测中的对应可选块应同步启用。

```bash
python scripts/supplement.py plan --suite mechanism_optional --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" --write-configs \
  | tee "$PLANS/mechanism_optional.json"
check_plan "$PLANS/mechanism_optional.json"
python scripts/supplement.py run-suite --suite mechanism_optional --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device "$DEVICE" "${EXECUTION_ARGS[@]}" \
  --execute --skip-completed | tee -a "$LOGS/mechanism_optional.log"
```

#### 4.2 单因素与第二协同骨干

此阶段必须已产生真实的 `$FINAL_MODEL`。rho/半径各 3 项默认只用 seed 17；数值影响实际计算，run_id 分别追加 `_rho0p125/0p25/0p5`、`_support_radius_m0p4/0p8/1p2`。它们用于开发分析，不根据最终测试重新决定 F。

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

第二骨干配置为 `$RUNS/configs/supp_v2xr_where2comm_{S0,F}_e40_s<seed>_native.yaml`；同 seed 公共参数初始化相同，保留通信选择和多尺度聚合。

#### 4.3 固定权重的开发诊断、干预与效率

```bash
# 每个 seed 均复用冻结结构对应的 selected；不使用 online 权重。
for S in 17 29 43; do
  ID="supp_v2xr_deer_sbr_${F_VARIANT}_e${EPOCHS}_s${S}"
  C="$RUNS/configs/$ID.yaml"
  CKPT="$RUNS/$ID/train/selected.pt"
  python scripts/supplement.py diagnose --config "$C" --checkpoint "$CKPT" \
    --split validate --weather normal --probe-condition all --samples 100 \
    --output "$RUNS/development/diagnostics/${ID}.json"
  python scripts/supplement.py robustness --config "$C" --checkpoint "$CKPT" \
    --suite interventions --split validate --weather normal --execute \
    --output "$RUNS/development/interventions/${ID}.json"
done

# BN 规模只在 train 重估、在开发集比较，不用测试选规模。
S=17
ID="supp_v2xr_deer_sbr_${F_VARIANT}_e${EPOCHS}_s${S}"
C="$RUNS/configs/$ID.yaml"
CKPT="$RUNS/$ID/train/selected.pt"
python scripts/supplement.py bn-sensitivity --config "$C" --checkpoint "$CKPT" \
  --split validate --weather normal --refit-samples 32 128 \
  --output "$RUNS/development/diagnostics/${ID}_bn.json"

# 同一 GPU、FP32、batch=1、相同开发样本分布下比较，另行记录 agent 分布。
for V in S0 "$F_VARIANT"; do
  ID="supp_v2xr_deer_sbr_${V}_e${EPOCHS}_s17"
  python scripts/supplement.py efficiency --config "$RUNS/configs/$ID.yaml" \
    --checkpoint "$RUNS/$ID/train/selected.pt" --split validate --weather normal \
    --samples 100 --warmup 20 --output "$RUNS/development/efficiency/${ID}.json"
done
```

诊断输出来自实际模型：类型 MAE/soft Brier、路由、供体/null、支撑和归一化幅度，四种探针共用开发场景。`--drop-fraction`、`--inject-fraction` 可逐强度另存。干预结果与重训消融分开。效率端到端包含 IO、预处理、传输、forward、解码、NMS，排除 AP/落盘；局部模块另次计时，CUDA 前后同步并释放跨帧张量。该独立效率入口显式关闭解析后raw cache，确保每帧仍读文件/解析；OS页缓存由系统管理，不声称冷盘IO。当前不统计真实通信包字节。

### 5. 真实数据扩展（数据和监督条件具备后）

本阶段仅索引服务器已有官方数据，不提供下载命令。它不阻塞 V2X-R 核心矩阵完成。两套真实数据输出目录独立于 `$RUNS`；使用同一个 `$FINAL_MODEL` 解析 F，不按目标测试成绩重选。

#### 5.1 V2X-Radar-C

需要官方 OPV2V 协同导出 `train/<sequence>/<agent>/<stamp>.yaml/.pcd/_radar.pcd` 和官方 `validate/`。正 ID 车辆为 ego，负 ID 为真实路侧；不得把模拟数据目录改名复用。官方数据入口见[数据卡](https://huggingface.co/datasets/yanglei18/V2X-Radar)和[协同子集目录](https://huggingface.co/datasets/yanglei18/V2X-Radar/tree/main/V2X-Radar-C)。

```bash
REAL=/data/v2x-radar-c
REAL_META=/data/metadata/v2xradar_supp_v1
REAL_RUNS=/data/runs/v2xradar_supp_v1
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

`prepare` 冻结 `$REAL_META/supplement_manifest.json`：从官方 train 留出 4 个完整 sequence 为开发，官方 validate 保留为最终 test 身份；可用 `--dev-sequences` 显式替代抽取。若有已审阅的 sequence→官方天气 JSON，可添加 `--weather-map`；实际计数、字段与标定以生成报告核验，不硬写样本数。

`CLEAR_SEQS=()` 表示没有自行声明晴天。真实 MDD-CORR 和 F 的类型监督需要已核验晴天 train 来源；未提供时上述整组预检会阻塞，不能继续执行其训练。晴天来源不足时可单独运行已有 S0/CoBEVT，以及下面的 L4DR 槽；不可用自然雨天作为 clean target。补充晴天来源时使用新 manifest/批次目录。MDD-CORR 只训练核验晴天子集，额外参考分支与训练池单列；严格匹配训练池时，其余配置也设 `data.clean_only: true` 并使用新 run_id。

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

主矩阵配置包括 `supp_v2xradar_deer_sbr_{S0,F}_e30_s<seed>.yaml`、`supp_v2xradar_mdd_S0_e30_s<seed>_native.yaml` 和 `supp_v2xradar_cobevt_S0_e40_s<seed>_native.yaml`。L4DR 槽为 `supp_v2xradar_l4dr_S0_e30_s<seed>_native.yaml`。输出均在 `$REAL_RUNS/<run_id>/train/`。官方 BEV 与项目 true-3D extension 分别命名，不把 extension AP 当作原论文官方成绩。

#### 5.2 K-Radar

需要已生成的官方 sparse Radar 产品、OS2-64 LiDAR、标定、description、v2.0 revised labels 和官方 train/test lists，不抽取或下载原始 tesseract。

```bash
KR=/data/kradar
KR_RADAR=/data/kradar_sparse_product
KR_LABEL=/data/KRadar_refined_label_by_UWIPL
KR_META=/data/metadata/kradar_supp_v1
KR_RUNS=/data/runs/kradar_supp_v1
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

`$KR/<seq>/os2-64/os2-64_<lidar_idx>.pcd`、`info_calib/calib_radar_lidar.txt`、`description.txt`，`$KR_RADAR/<seq>/sprdr_<radar_idx>.npy` 和 `$KR_LABEL/<seq>/<label_name>.txt` 由 label header 对齐，不能只匹配同名文件。官方 description 的 `normal` 用于核验晴天，MDD 没有合格 train 参考时会阻塞。

配置为 `supp_kradar_deer_sbr_{S0,F}_e30_s<seed>.yaml`、`supp_kradar_mdd_S0_e30_s<seed>_native.yaml`、`supp_kradar_l4dr_S0_e35_s<seed>_native.yaml`。L4DR 保留 35 轮 AdamW/cosine、batch=3 候选，不强改成 30 轮。所有输出在 `$KR_RUNS/<run_id>/train/`。这是实际单车、双类别 AP41，不复制帧制造 agent；只保证 train/development sequence 隔离，官方最终 test 帧和其 train/test sequence 交叠身份保留。坐标、ROI、类别、输入字段与 evaluator 依据见[正式方案](13_补充实验方案与实施说明.md#81-实现中冻结的关键语义)和[官方评分源说明](../opencood/project/official/README.md)。

### 6. 最终评测、鲁棒性与统计汇总

#### 6.1 用各 seed 的 selected 完整评测

只有当训练预算、结构、BN 规则、后处理和实验矩阵已由开发证据冻结后，执行本节。每份配置只使用其对应 selected；三个天气共用同一权重。普通评测覆盖该 split 的全部样本；最终测试均显式传 `--allow-test-evaluation`。

以下函数读取已保存的任务清单，自动取得准确的配置、selected 和数据集身份。V2X-R 依次评测 Normal/Fog/Snow，真实数据使用 `natural`；不从文件名推断模型，也不把自然天气改为模拟 Fog。

```bash
FINAL_ROOT="$RUNS/final_eval/selected_v1"
evaluate_plan() {
  local plan="$1" destination="$2"
  local run_id config checkpoint dataset weather
  local -a weathers
  while IFS=$'\t' read -r run_id config checkpoint dataset; do
    weathers=(normal fog snow)
    if [[ "$dataset" != v2xr ]]; then weathers=(natural); fi
    for weather in "${weathers[@]}"; do
      python scripts/supplement.py evaluate --config "$config" --checkpoint "$checkpoint" \
        --split test --weather "$weather" --allow-test-evaluation \
        --output "$destination/$run_id/${weather}.scores.json"
    done
  done < <(python -c \
    'import json,sys; [print("\t".join([t["run_id"],t["config"],t["main_outputs"][1],t["dataset"]])) for t in json.load(open(sys.argv[1]))["tasks"]]' \
    "$plan")
}

# 内部 18 项 × 3 天气 = 54 次，外部 12 项 × 3 天气 = 36 次完整测试。
evaluate_plan "$PLANS/internal_main.json" "$FINAL_ROOT/internal"
evaluate_plan "$PLANS/external_main.json" "$FINAL_ROOT/external"
# 机制核心 9 项 × 3 天气 = 27 次，与核心主矩阵分目录保存。
evaluate_plan "$PLANS/mechanism_pilot.json" "$FINAL_ROOT/mechanism"
evaluate_plan "$PLANS/mechanism_repeat.json" "$FINAL_ROOT/mechanism"
```

仅对已实际训练完成的扩展块执行其对应评测，不为未运行实验补结果：

```bash
evaluate_plan "$PLANS/mechanism_optional.json" "$FINAL_ROOT/mechanism_optional"
evaluate_plan "$PLANS/second_backbone.json" "$FINAL_ROOT/second_backbone"
evaluate_plan "$REAL_RUNS/plans/real_cooperative.json" "$REAL_RUNS/final_eval/selected_v1/main"
evaluate_plan "$REAL_RUNS/plans/real_cooperative_l4dr.json" "$REAL_RUNS/final_eval/selected_v1/l4dr"
evaluate_plan "$KR_RUNS/plans/real_weather.json" "$KR_RUNS/final_eval/selected_v1/main"
```

一项评分路径例如：

```text
$RUNS/final_eval/selected_v1/internal/supp_v2xr_deer_sbr_AB_e60_s17/
  snow.scores.json                       # AP 与完整协议/权重/GT 元数据
  snow.scores.predictions.jsonl          # 相同 selected 的完整预测
  snow.scores.predictions.jsonl.meta.json
```

真实数据评分另包含官方自然天气标签和距离组；V2X-Radar-C 官方 validate 在准备时映射成这里的 `--split test`。不在测试阶段重估 BN。若显式研究预算快照，应单独指定 `budget_epochNNN.pt`、`--budget-snapshot` 和独立输出目录，不与 selected 主表混算。

rho/半径默认只做开发筛查。完成训练后这样生成开发评分；如预先登记了其最终测试，再使用同样的 `evaluate` 命令与显式 test 开关，单独保存，不能利用测试结果重新调参。

```bash
for SUITE in rho radius; do
  while IFS=$'\t' read -r ID C CKPT; do
    for W in normal fog; do
      python scripts/supplement.py evaluate --config "$C" --checkpoint "$CKPT" \
        --split validate --weather "$W" \
        --output "$RUNS/development/$SUITE/$ID/${W}.scores.json"
    done
  done < <(python -c \
    'import json,sys; [print("\t".join([t["run_id"],t["config"],t["main_outputs"][1]])) for t in json.load(open(sys.argv[1]))["tasks"]]' \
    "$PLANS/$SUITE.json")
done
```

#### 6.2 鲁棒性矩阵与固定权重干预

在 Normal 输入上，S0 与冻结 F 各三 seed 使用相同 suite、样本和 `perturbation_seed=17029`。下块运行真实推理，默认包含删点/注入/共存、Radar 保留率与模态缺失、agent 数、异质 agent、定位及异步条件；从头训练消融和第 4.3 节开发干预分别保留。

```bash
for V in S0 "$F_VARIANT"; do
  for S in 17 29 43; do
    ID="supp_v2xr_deer_sbr_${V}_e${EPOCHS}_s${S}"
    for SUITE in corruption radar agents heterogeneous pose async; do
      python scripts/supplement.py robustness --config "$RUNS/configs/$ID.yaml" \
        --checkpoint "$RUNS/$ID/train/selected.pt" --suite "$SUITE" \
        --split test --weather normal --allow-test-evaluation --execute \
        --output "$FINAL_ROOT/robustness/$ID/${SUITE}.json"
    done
  done
done
```

各 suite 保存 `<suite>.common_samples.json`，条件评分在 `<suite>/<condition>/result.scores.json`。使用同一数据、同一 suite 时保留共同样本和扰动身份；汇总会拒绝方法间样本/GT 不一致。agent 数只统计真实可通信 agent，不能满足的数量标为 `not_applicable`；async 使用真实历史帧、当前 GT，延迟为帧数。K-Radar 不运行协同定位、异步、异质 agent 或伪造多 agent。

#### 6.3 主表、消融、第二骨干与配对统计

```bash
python scripts/supplement.py summarize \
  --results "$FINAL_ROOT/internal" "$FINAL_ROOT/external" \
  --baseline deer_sbr/S0 --output "$REPORTS/main_selected.json"
python scripts/supplement.py summarize \
  --results "$FINAL_ROOT/internal" "$FINAL_ROOT/mechanism" \
  --baseline deer_sbr/S0 --output "$REPORTS/mechanism_selected.json"
# 每个 suite 单独汇总，避免把多个 suite 的相同 baseline 重复作为一个 seed。
for SUITE in corruption radar agents heterogeneous pose async; do
  ROBUST_PATHS=()
  for V in S0 "$F_VARIANT"; do
    for S in 17 29 43; do
      ID="supp_v2xr_deer_sbr_${V}_e${EPOCHS}_s${S}"
      ROBUST_PATHS+=("$FINAL_ROOT/robustness/$ID/$SUITE")
    done
  done
  python scripts/supplement.py summarize --results "${ROBUST_PATHS[@]}" \
    --baseline deer_sbr/S0 --output "$REPORTS/robustness_${SUITE}_selected.json"
done

# sequence bootstrap 示例：seed 17、Snow、S0 与唯一 F。
S=17
BASE_ID="supp_v2xr_deer_sbr_S0_e${EPOCHS}_s${S}"
F_ID="supp_v2xr_deer_sbr_${F_VARIANT}_e${EPOCHS}_s${S}"
python scripts/supplement.py bootstrap --config "$RUNS/configs/$BASE_ID.yaml" --split test \
  --predictions-a "$FINAL_ROOT/internal/$BASE_ID/snow.scores.predictions.jsonl" \
  --predictions-b "$FINAL_ROOT/internal/$F_ID/snow.scores.predictions.jsonl" \
  --repetitions 1000 --output "$REPORTS/paired_sequence_s${S}_snow.json"
```

以下汇总与已执行的可选块对应；真实数据分开输出，不混入模拟天气均值。V2X-Radar-C 仅完成主矩阵时，汇总中省去 L4DR 目录参数：

```bash
python scripts/supplement.py summarize \
  --results "$FINAL_ROOT/internal" "$FINAL_ROOT/mechanism_optional" \
  --baseline deer_sbr/S0 --output "$REPORTS/mechanism_optional_selected.json"
python scripts/supplement.py summarize --results "$FINAL_ROOT/second_backbone" \
  --baseline where2comm/S0 --output "$REPORTS/second_backbone_selected.json"
python scripts/supplement.py summarize \
  --results "$REAL_RUNS/final_eval/selected_v1/main" "$REAL_RUNS/final_eval/selected_v1/l4dr" \
  --baseline deer_sbr/S0 --output "$REPORTS/v2xradar_selected.json"
python scripts/supplement.py summarize --results "$KR_RUNS/final_eval/selected_v1/main" \
  --baseline deer_sbr/S0 --output "$REPORTS/kradar_selected.json"
for SUITE in rho radius; do
  python scripts/supplement.py summarize --results "$RUNS/development/$SUITE" \
    --output "$REPORTS/${SUITE}_development.json"
done
```

`convergence` 输出 JSON/CSV/PNG，`summarize` 输出 JSON/CSV；AP 原始数值为 0–1，CSV 的 `mean_percent` 和 `sample_std_pp` 为百分数/百分点。逐 seed 配对，样本标准差使用 ddof=1；Fog/Snow 先在同 seed、同 selected 内取均值再跨 seed 汇总。种子不足明确标记 `complete_three_seeds=false`，不补齐。rho/半径、不同预算、预算快照、不同 transfer 身份单独分组；单因素目录没有 S0 时仅报告参数组成绩，不生成配对基线差。多个 baseline 预算不自动择优混算。sequence bootstrap 成对重采样并重算 AP，与训练种子方差分别解释。

固定场景图可在已评测预测上运行 `visualize`，输出 1–5 个真实观测/GT/预测 BEV 场景；不重新评分：

```bash
python scripts/supplement.py visualize --config "$RUNS/configs/$F_ID.yaml" \
  --predictions "$FINAL_ROOT/internal/$F_ID/snow.scores.predictions.jsonl" \
  --output "$RUNS/visualizations/${F_ID}_snow.json"
```

HRCP、REL 当前没有可核查完整适配，不提供虚构运行命令。上述命令在真实执行前不构成新增 AP、收敛或资源成绩。所有实际本地检查及未验证路径统一见 [VALIDATION.md](VALIDATION.md)。

### 7. Windows无真实数据检查

本地检查使用UTF-8。若Anaconda的NumPy MKL与PyTorch出现OpenMP #15冲突，可在该检查进程选择sequential MKL；该设置不用于论文效率测量：

```powershell
$env:PYTHONUTF8 = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python -m unittest tests.test_supplement tests.test_supplement_integrity tests.test_l4dr_supplement -v
python scripts/supplement.py smoke --output outputs/supplement_cpu_check.json
python scripts/reproduce.py results
```

此处smoke仅运行合成前后向与推理，不更新优化器。没有真实数据时，native数据读取、训练和最终评分仍需在已准备数据及扩展的环境验证。实际执行模块与缺数据skip详见[VALIDATION.md](VALIDATION.md)。
