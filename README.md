# CSCWD · DEER + SBR

V2X-R LiDAR–Radar 协同三维检测的可复现代码：**DEER**（缺失/伪回波证据路由）与 **SBR**（真实观测支撑的有界残差修复）。基于 [V2X-R](https://github.com/ylwhxht/V2X-R) / OpenCOOD，包含 S0、A、B、AB、SUP、N1、N2 七组对照。

本仓库复现的是实际完成的 **teacher_free_v4、seed=17、20 epoch** 实验。每组有效 batch=8、FP32、单 GPU；Normal/Fog 验证选模，Snow 仅用于模型冻结后的最终测试。历史 30 epoch 计划和未执行的外部基线不属于已完成结果。

## 快速开始

环境：Linux x86_64、Python 3.10、NVIDIA GPU、CUDA 12.1 开发工具链（含 `nvcc`）及 GCC/G++。原实验使用 RTX 2080 Ti，服务器显示每卡约 22GB 显存；普通 11GB 版本需要单独验证内存预算。预留至少约 220GiB 数据盘空间，下载并行度较高时需要更多空间。

```bash
sudo apt-get update
sudo apt-get install -y git gh
# 私有仓库需要有访问权限的 GitHub 账号
gh auth login
gh auth setup-git
git clone https://github.com/Hqa77880011/CSCWD-DEER-SBR.git
cd CSCWD-DEER-SBR
conda env create -f environment.yml
conda activate cscwd-repro
sudo apt-get install -y build-essential aria2 p7zip-full libgl1 libglib2.0-0
bash scripts/install.sh
```

下载固定版本的数据、保留点云与标注、仅读取 PNG 尺寸元数据，并生成固定 GT：

```bash
python scripts/prepare_data.py --data-root "$PWD/datasets/v2xr" --workers 2
```

## 下载训练权重，直接复现评测

七组原始 `selected.pt` 已提供于 [v1.0.0 Release](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/tag/v1.0.0)，每份约 302MB。文件与正式测试使用的 checkpoint 逐字节一致，包含所选权重及冻结的 train-only BN 统计。共享评测元数据约 51MiB；[SHA256SUMS](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/SHA256SUMS) 提供下载校验。

| S0 | A | B | AB | SUP | N1 | N2 |
|---|---|---|---|---|---|---|
| [权重](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/S0-selected.pt) | [权重](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/A-selected.pt) | [权重](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/B-selected.pt) | [权重](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/AB-selected.pt) | [权重](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/SUP-selected.pt) | [权重](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/N1-selected.pt) | [权重](https://github.com/Hqa77880011/CSCWD-DEER-SBR/releases/download/v1.0.0/N2-selected.pt) |

完成上述环境安装和数据准备后，可直接评测，无需训练或重新拟合 BN。仓库为私有，需要以有访问权限的 GitHub 账号登录 `gh`：

```bash
gh auth login
# 下载 AB 原始权重及共享评测元数据，自动校验 SHA256
python scripts/pretrained.py download --variant AB
# 直接重现 AB 的完整三天气测试（每项 3166 样本、30296 GT）
CUDA_VISIBLE_DEVICES=0 python scripts/pretrained.py evaluate \
  --data-root /your/path/v2xr --variant AB --weather all --allow-test-evaluation
```

只评测 AB 的 Snow，预期 AP3D@0.7 约为 49.19%：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/pretrained.py evaluate \
  --data-root /your/path/v2xr --variant AB --weather snow \
  --output-root outputs/ab_snow --allow-test-evaluation
```

下载并评测全部七组、三天气（约 2.02GiB 下载，21 项完整测试，耗时较长）：

```bash
python scripts/pretrained.py download --variant all
CUDA_VISIBLE_DEVICES=0 python scripts/pretrained.py evaluate \
  --data-root /your/path/v2xr --variant all --weather all --allow-test-evaluation
```

结果写入 `outputs/pretrained_evaluation/<variant>/<weather>/`，含预测、完整 AP3D/APBEV 和 `summary.json`，并报告与原表 AP3D@0.7 的差值。若目录已有结果，请指定新的 `--output-root`。数据目录可更换；工具使用 Release 中原始元数据建立独立视图，不覆盖你的数据清单，仍核对原 checkpoint、科学源码、配置、读取的点云/标注及冻结 BN。跨 GPU/工具链可能产生数值差异。更多说明见 [复现指南](docs/REPRODUCING.md#使用-release-权重直接评测)。

## 从头训练

已有本项目准备好的数据（必须包含 `data_manifest.json`、`eval_gt_manifest.json` 等元数据）可直接配置路径：

```bash
python scripts/reproduce.py configure --data-root /your/path/v2xr
python scripts/reproduce.py audit
python scripts/reproduce.py preflight --variant all
python scripts/reproduce.py smoke --variant AB
bash scripts/test.sh
```

从共同种子初始化重训七组（按顺序运行，耗时较长）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py train --variant all
```

训练完成、方法冻结后执行三天气测试：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py evaluate --variant all --allow-test-evaluation
```

详细安装、数据恢复、断点续训、选模规则与复现边界见 [复现指南](docs/REPRODUCING.md)。

## 已有结果

单种子、20 epoch，Vehicle AP3D@IoU=0.7（%）；每项测试覆盖 3166 个样本、30296 个 GT。

| 组别 | Normal | Fog | Snow | Fog/Snow 均值 |
|---|---:|---:|---:|---:|
| S0 | 65.46 | 51.25 | 45.67 | 48.46 |
| A | 64.67 | 51.60 | 46.22 | 48.91 |
| B | 65.18 | 51.41 | 44.72 | 48.06 |
| AB | 65.19 | 51.07 | **49.19** | 50.13 |
| SUP | 64.84 | 51.36 | 46.24 | 48.80 |
| N1 | 64.37 | 51.03 | 48.69 | 49.86 |
| N2 | **65.87** | **52.52** | 48.73 | **50.62** |

AB 相对 S0 的 Snow 增益为 3.52 个百分点，天气均值增益为 1.67 个百分点；Normal/Fog 略降，N2 的天气均值更高。这些是内部对照结果，尚无多种子统计或同协议外部 SOTA 对比。

完整原值、三个 IoU 阈值的 AP3D/APBEV、所选权重哈希和训练/测试 manifest 位于 [`results/reference/`](results/reference)。可以离线核算表格：

```bash
python scripts/reproduce.py results
```

该命令检查已有 CSV 的计算一致性；重新测得 AP 可用上面的 Release 权重直接评测，也可从头训练后评测。

## 目录

```text
opencood/             模型、数据集、训练协议、固定 GT 和评测实现
project_tools/        原始训练、预测导出、评测与诊断工具
scripts/              可移植的安装、数据准备及七组复现入口
repro/configs/        实际执行的七组 20 epoch 配置快照
repro/metadata/       固定数据版本、源文件哈希、划分与 GT 摘要
results/reference/   原始结果表与实验身份记录
tests/               协议、模块、数据几何、评测及复现入口回归
docs/                当前复现指南及历史设计/实验分析
```

数据、Python 环境、CUDA 编译产物、预测归档和大模型权重不存入 Git；训练权重与原始评测元数据存放在 GitHub Release。历史文档和 manifest 中的服务器路径仅用于追溯；从头训练使用 `configure` 生成的新路径，已有权重评测使用 `scripts/pretrained.py`。

上游说明保留在 [`readme.md`](readme.md)。作者署名和第三方许可说明见 [NOTICE](NOTICE.md)。
