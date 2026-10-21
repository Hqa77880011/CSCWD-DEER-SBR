# CSCWD · DEER + SBR

V2X-R LiDAR–Radar 协同三维检测的可复现代码：**DEER**（缺失/伪回波证据路由）与 **SBR**（真实观测支撑的有界残差修复）。基于 [V2X-R](https://github.com/ylwhxht/V2X-R) / OpenCOOD，包含 S0、A、B、AB、SUP、N1、N2 七组对照。

本仓库复现的是实际完成的 **teacher_free_v4、seed=17、20 epoch** 实验。每组有效 batch=8、FP32、单 GPU；Normal/Fog 验证选模，Snow 仅用于模型冻结后的最终测试。历史 30 epoch 计划和未执行的外部基线不属于已完成结果。

## 快速开始

环境：Linux x86_64、Python 3.10、NVIDIA GPU、CUDA 12.1 开发工具链（含 `nvcc`）及 GCC/G++。原实验使用 RTX 2080 Ti，服务器显示每卡约 22GB 显存；普通 11GB 版本需要单独验证内存预算。预留至少约 220GiB 数据盘空间，下载并行度较高时需要更多空间。

```bash
git clone https://github.com/Hqa77880011/CSCWD-DEER-SBR.git
cd CSCWD-DEER-SBR
conda env create -f environment.yml
conda activate cscwd-repro
sudo apt-get update
sudo apt-get install -y build-essential git aria2 p7zip-full libgl1 libglib2.0-0
bash scripts/install.sh
```

下载固定版本的数据、保留点云与标注、仅读取 PNG 尺寸元数据，并生成固定 GT：

```bash
python scripts/prepare_data.py --data-root "$PWD/datasets/v2xr" --workers 2
```

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

该命令检查已有 CSV 的计算一致性；重新测得 AP 需要完整数据和重新训练。

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

数据、Python 环境、CUDA 编译产物、预测归档和大模型权重不存入 Git。仓库提供从头训练流程；当前未提供已训练权重的外部下载链接。历史文档和 manifest 中的服务器路径仅用于追溯；运行时使用 `configure` 生成的新路径。

上游说明保留在 [`readme.md`](readme.md)。作者署名和第三方许可说明见 [NOTICE](NOTICE.md)。
