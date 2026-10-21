# 复现指南

## 固定的实验身份

实际实验为 V2X-R、`teacher_free_v4` / schema 4、`project_fix_v1` 评分协议、W 训练线、seed 17、20 epoch。七组为 S0/A/B/AB/SUP/N1/N2，各 20200 次有效优化更新；源数据 revision 为 `4c9ff1773b49517464714368d30778b5252bdbd0`。原始配置保存在 `repro/configs/`，训练和测试身份保存在 `results/reference/`。

原批次中 S0 从已核验的第 1000 步 online checkpoint 续训，其累计预算为 20 epoch。这里从同一种子重新初始化并完整训练，不依赖该服务器 checkpoint。CUDA/GPU/编译器差异可能造成数值差异；不承诺跨硬件逐位相同或 AP 精确到最后一位。

## 环境

从仓库根目录执行 README 中的 Conda 和安装命令。`requirements-repro.txt` 固定主要依赖，`requirements-repro-lock.txt` 记录本次服务器环境中这些依赖的传递依赖版本。安装脚本默认使用该文件作为 constraints 固定传递依赖。上游 `requirements.txt` 与 `INSTALL.md` 是历史配方。

PyTorch 为 2.1.2+cu121、torchvision 为 0.16.2+cu121、NumPy 为 1.26.3，spconv/cuMM 为 2.3.6/0.4.11。spconv wheel 名称中的 cu120 是已有环境使用的发布包名称。完整 CUDA toolkit 需提供 `nvcc`；驱动由宿主机提供。`project_tools/build_extensions.sh` 自动读取可见 GPU 架构。交叉编译时显式设置，例如 `TORCH_CUDA_ARCH_LIST=7.5`。

`bash scripts/install.sh` 构建 Cython 与 CUDA 扩展，执行 `pip check`，并运行合成 GPU 前向/反向、IoU/NMS 和 checkpoint 检查。详细结果写入 `outputs/environment_check.json`。环境安装不会启动正式训练。

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

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py evaluate --variant all \
  --output-root /disk/runs/seed17 --allow-test-evaluation
```

该命令分别导出每组 Normal/Fog/Snow 的全部测试样本，并调用同一个独立评测器。预测 JSONL、哈希 sidecar 和结果 JSON 位于 `<output-root>/final_test/<variant>/`。有同名输出时会停止，避免意外覆盖已冻结结果；失败后请先检查该目录，使用原始 `export_predictions.py` / `evaluate_predictions.py` 定位恢复。

固定 GT 使用原始标注与标定，ego 中心 ROI 为 x∈[0,140)、y∈[-40,40)，固定 camera0 几何 FOV；不按天气点数删 GT，不截断 GT。评分是全局排序的 VOC2010 全点插值 AP，置信度下限 0.001、BEV NMS IoU=0.15、每样本最多 500 个预测。三天气共用同一组选中权重。测试为 3166 样本、30296 GT，天气均值为 Fog/Snow AP3D@0.7 算术均值。

`python scripts/reproduce.py results` 仅复核已有结果表的均值和增益。现有仓库没有上传原始数据、数十 GB 训练输出或预训练权重；从头训练可重建可评测 checkpoint。原始结果 manifest 中的绝对路径和旧 SHA256 是历史证据；迁移路径后新实验会生成新的文件身份，不能拿历史权重绕过配置/来源校验。

## 验证

```bash
bash scripts/test.sh
# 若已准备全量数据，可额外启用真实 train 样本/固定 GT 回归
CSCWD_DATA_ROOT=/disk/v2xr bash scripts/test.sh
```

默认回归使用合成输入；依赖全量数据的测试在数据不可用时明确 skip。`ci/github-actions.yml.example` 提供 GitHub Actions 模板，覆盖静态编译、结果核算与可移植入口回归，不提供 GPU。当前登录的 GitHub 凭据具有 repo 权限、没有 workflow 权限，因此模板未激活；需要时自行移至 `.github/workflows/checks.yml` 并以具备 workflow 权限的凭据推送。当前机器上实际做过的检查见 [验证记录](VALIDATION.md)。

历史中文设计和分析文档用于追溯。含旧 30 epoch、旧版本或服务器绝对路径的内容，以本指南和 `repro/configs/` 的 20 epoch 实际配置为准。结果局限包括单种子、单数据集、内部对照及未经测量的端到端资源成本。
