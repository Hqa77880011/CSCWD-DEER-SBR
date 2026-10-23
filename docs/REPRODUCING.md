# 复现指南

原七组 20 轮结果使用下文原复现入口；新增实验使用本文末尾的 schema 5 入口。当前补充实验方案以 [13_补充实验方案与实施说明.md](13_补充实验方案与实施说明.md) 为准。新代码准备完成不代表实验完成。

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

所有命令从仓库根目录执行，以下为 Linux / Bash。先使用本文已有环境安装流程；本节不会自动下载数据或预训练权重。原 `scripts/reproduce.py` 与 `scripts/pretrained.py` 仍执行原七组代码：源码来自 `repro/archive/legacy_v4_runtime.zip`，原配置/结果/权重身份验证保留，并使用服务器在主仓库编译的扩展二进制；默认数据、输出和权重目录仍位于主仓库。发布helper已修复Windows临时文件关闭后的替换及无符号链接权限时的junction/硬链接数据视图；原科学源码、七份配置及Release权重身份不变。请将该小型源码包一并上传，不能被 `*.zip` 的忽略规则漏掉。

### 1. 环境、数据与独立配置

```bash
export PYTHONUTF8=1
DATA=/data/v2xr                 # 已准备的 V2X-R 原点云及固定 metadata
RUNS=/data/runs/supplement_v1    # 全新输出目录
EPOCHS=60                       # 开发曲线确定后可改 20 / 30 / 60 / 90

python scripts/supplement.py environment --native-v2xr --device cuda:0
python scripts/supplement.py smoke --output "$RUNS/synthetic_smoke.json"

# 查看任务并生成独立完整配置；不启动实验
python scripts/supplement.py plan --suite pilot --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device cuda:0 --write-configs
C="$RUNS/configs/supp_v2xr_deer_sbr_AB_e${EPOCHS}_s17.yaml"
python scripts/supplement.py preflight --config "$C"
python scripts/supplement.py check-data --config "$C" --samples 2
```

`environment` 报告依赖缺失与设备状态；`--native-v2xr` 还检查原 V2X-R loader 的 spconv/cuMM、open3d、pypcd 和扩展导入。它不把导入检查称为 CUDA 数值验证。`smoke` 在 CPU 合成小网格上保持原生层数/通道、MDD 扩散步数和 loss，进行前向/反向/推理，零优化更新、不读取真实数据、不输出 AP。CoBEVT 检查非窗口整除形状，MDD 检查单 agent，K-Radar 检查双类别。

V2X-R 沿用上文原数据准备流程；本轮实现时没有调用下载脚本；prepare格式测试仅使用合成文件样例。新 preflight 的 `formal_blockers` 必须为空；`check-data` 会实际读取少量 train 样本，核对输入、正 anchor、agent 和点数，不训练。路径不存在时明确报错，没有合成数据回退。`--device` 可用 `auto`、`cpu`、`cuda:0` 等；数据、manifest、输出根通过 configure/plan 传入。完整 GPU 环境的旧检查仍可在服务器运行 `python project_tools/check_environment.py --output "$RUNS/environment_full.json"`。

### 2. 单项、先导、多种子与恢复

```bash
# 单项：只有显式 --execute 才训练
python scripts/supplement.py train --config "$C" --execute

# 先导三组；已完成且配置相同的单项可显式跳过
python scripts/supplement.py run-suite --suite pilot --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device cuda:0 --execute --skip-completed
python scripts/supplement.py run-suite --suite pilot_sup --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device cuda:0 --execute

# 核心内部组 × 三种子；省去 --execute 时只展开任务
python scripts/supplement.py run-suite --suite internal_main --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device cuda:0 --execute --skip-completed

# 外部完整方法保留原生 batch（AttFuse=8，K-Radar L4DR=3，其余已接候选=2）/ optimizer / schedule；CoBEVT / Where2comm 原生 40 轮不强改为 60
python scripts/supplement.py plan --suite external_main --data-root "$DATA" \
  --output-root "$RUNS" --device cuda:0
python scripts/supplement.py run-suite --suite external_main --data-root "$DATA" \
  --output-root "$RUNS" --device cuda:0 --execute --skip-completed

# online 恢复，不接受 selected / budget snapshot 作为 resume
python scripts/supplement.py train --config "$C" --execute \
  --resume "$RUNS/supp_v2xr_deer_sbr_AB_e${EPOCHS}_s17/train/latest.rank0.pt"
```

所有新训练仍调用 `project_tools/run_experiments.py`。也可直接使用 `--mode train --allow-formal-run`；该底层入口支持 `--initialize-from`，仅供登记为 `experiment.transfer: fine_tuned` 的新配方，不迁移优化器/RNG。`zero_shot` 配置禁止目标训练；在同类别/同结构的 schema 5 源权重上独立评测，不拟合目标 BN。当前没有自动把 V2X-R 单类权重变成 K-Radar 双类权重的路径。

一次任务只用一张 GPU 时，可通过 `CUDA_VISIBLE_DEVICES` 分配不同终端；不自动连接服务器或调度多台机器。`run-suite` 顺序执行，`--seeds 29 43` 可只展开指定种子；不要把单任务多卡与独立多任务混称模型推理加速。

每组 `<RUNS>/<run_id>/train/` 内保存：

- `resolved_config.json`、`run_metadata.json`、`steps.rank0.jsonl`：有效 batch、累计更新、实际 lr、loss 各项、输入天气/点数与额外参考成本。
- `validation_epochNNN.json`：每个注册开发域的完整评分；`state.json` 记录所选轮次。
- `latest.rank0.pt`：online BN + 优化器/调度器/采样器/RNG，仅 resume。
- `selected.pt`：按开发指标选择的权重；V2X-R 内部组用 train-only BN 重估，完整外部方法保留 native running 统计；真实数据初始候选配方使用 running，避免未经验证地照搬模拟数据的 32 样本重估规模。
- `budget_epochNNN.pt`：当前轮权重及其评测 BN，明确标为 `long_recipe_intermediate` 或 `independent_recipe_endpoint`；不会把 60 轮计划的第 30 轮自动改称独立 30 轮配方。

若训练预算要改变，请重新 `configure` 或 `plan --epochs` 生成新 run_id；不把旧 20 轮 resume 伪装成从头 60 轮。新日志能直接进入 `convergence`，不使用测试曲线判断平台。

### 3. 冻结唯一 F、消融与单因素

```bash
DEV_RUNS=()
for V in S0 AB N2; do
  for S in 17 29 43; do
    DEV_RUNS+=("$RUNS/supp_v2xr_deer_sbr_${V}_e${EPOCHS}_s${S}/train")
  done
done
python scripts/supplement.py freeze-final --run-dirs "${DEV_RUNS[@]}" \
  --normal-tolerance-pp 1 --output "$RUNS/frozen_final.json"

python scripts/supplement.py run-suite --suite mechanism_pilot --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device cuda:0 --execute
python scripts/supplement.py run-suite --suite mechanism_repeat --epochs "$EPOCHS" \
  --data-root "$DATA" --output-root "$RUNS" --device cuda:0 --execute
python scripts/supplement.py plan --suite rho --data-root "$DATA" \
  --output-root "$RUNS" --final-model-file "$RUNS/frozen_final.json" --write-configs
python scripts/supplement.py plan --suite radius --data-root "$DATA" \
  --output-root "$RUNS" --final-model-file "$RUNS/frozen_final.json" --write-configs
```

冻结工具只接受 S0/AB/N2 × 17/29/43 的匹配完整开发日志与对应 selected 权重，Normal 容忍线必须等于训练配置中事先登记的 `selection.normal_tolerance_pp`。它输出一个 AB/N2 结构及权重证据，不逐天气/seed 选测试赢家。V2X-R 最终模型可复用记录中同 seed 的候选 selected，不要求再训练一次 F。rho 与 radius suite 默认 60 轮；已冻结其他预算时同样可传 `--epochs`。M1–M6 / NO_NULL 当前是 AB 基础的从头训练控制；若 F 为 N2，不把这些控制的解释自动扩展到 N2。

`mechanism_optional` 包含 M2/M5/M6/N1/NO_NULL，仍只按方案需要执行。第二协同骨干使用 `second_backbone --final-model-file ...`，底座及 +F 共同保留 Where2comm 的通信选择与多尺度聚合。完整方法与公共底座模块比较的身份在配置/结果中单独记录。真实数据suite的正式名称为`real_cooperative`、`real_cooperative_l4dr`、`real_weather`，旧CLI别名仍兼容；原生配方run_id增加`_native`后缀，避免与同轮数独立配方碰撞。

### 4. 选中权重、鲁棒性、诊断与效率

```bash
CKPT="$RUNS/supp_v2xr_deer_sbr_AB_e${EPOCHS}_s17/train/selected.pt"

# 开发评测；最终测试还需显式 --split test --allow-test-evaluation
python scripts/supplement.py evaluate --config "$C" --checkpoint "$CKPT" \
  --split validate --weather normal --output "$RUNS/eval/AB_s17_normal.scores.json"

# 仅预览；加 --execute 才运行推理，不重新训练
python scripts/supplement.py robustness --config "$C" --checkpoint "$CKPT" \
  --suite corruption --split validate --output "$RUNS/robustness/corruption_plan.json"
python scripts/supplement.py robustness --config "$C" --checkpoint "$CKPT" \
  --suite radar --split validate --output "$RUNS/robustness/radar.json" --execute

python scripts/supplement.py diagnose --config "$C" --checkpoint "$CKPT" \
  --split validate --weather normal --probe-condition all --samples 100 \
  --output "$RUNS/diagnostics/AB_s17.json"
python scripts/supplement.py efficiency --config "$C" --checkpoint "$CKPT" \
  --split validate --weather normal --samples 100 --warmup 20 \
  --output "$RUNS/efficiency/AB_s17.json"
```

鲁棒性 suite 为 `corruption`、`radar`、`agents`、`heterogeneous`、`pose`、`async`、`interventions`。`--execute` 前建立条件间的公共帧交集，并保存 `*.common_samples.json`；各方法应使用同一数据、相同 suite 和固定 `perturbation_seed`。最终预测记录保留实际 agent、source timestamp、投影矩阵和删点/注入来源追踪。`agents` 在实际可用的共同场景上做 ego-only/2/3/5，不能满足的数量明确标为 not_applicable。K-Radar 不运行协同定位/延迟/异质 agent 试验。

`diagnose --probe-condition all` 在相同独立开发样本上生成无新增扰动、只删点、只注入、共存四种输入，报告连续标签 MAE / soft Brier、路由增益、null、支撑内外残差、归一化幅度、供体RMS/余弦及raw/tanh统计。可用`--probe-condition drop --drop-fraction 0.2`等生成逐强度独立报告，注入探针使用`--inject-fraction`。真实数据类型精度只使用核验晴天的开发场景；未知自然天气不造类型 GT。`interventions` 固定同一权重关闭或打乱类型/幅度/供体信号，结果与重训消融分开。GT 与 Radar 保留率、模态缺失、agent cap 和当前点数无关。

`efficiency` 的参数量、端到端峰值 allocated 显存、端到端/局部 p50/p95 分开记录。端到端串行包含点文件 IO、预处理、传输、forward、解码、NMS，排除评分/落盘；局部另一次计时仅包含 phi/DEER/SBR。CUDA 前后同步。`bn-sensitivity --config "$C" --checkpoint "$CKPT" --refit-samples 32 128 --output "$RUNS/diagnostics/bn.json"` 可固定内部权重，只在 train 重估后比较开发成绩，零优化更新并恢复原 BN；不得用测试成绩确定规模。CPU 没有伪造 GPU 显存，正式成本比较必须在同 GPU/精度/batch/agent 分布上测量。没有统计真实通信包字节，不据此声称低通信。

普通评测覆盖配置选定 split 的所有样本；只接受 selected，预算快照需额外 `--budget-snapshot`。输出 `<name>.predictions.jsonl`、metadata sidecar、`*.scores.json`，真实数据另含官方天气标签和距离组。自然数据只接受 `--weather natural`；不能通过 `--weather fog` 为真实数据伪造 Fog 文件。

### 5. 真实数据准备及运行

V2X-Radar-C发布入口见[官方数据卡](https://huggingface.co/datasets/yanglei18/V2X-Radar)及[协同子集目录](https://huggingface.co/datasets/yanglei18/V2X-Radar/tree/main/V2X-Radar-C)，容量和许可按该版本核实；代码只索引本地文件。V2X-Radar-C 必须是官方 OPV2V 协同导出：`train/<sequence>/<agent>/<stamp>.yaml/.pcd/_radar.pcd` 及官方 `validate/`。正 ID 车辆为 ego，负 ID 为真实路侧；不同传感器字段按数据集分别读取。不要将目录改名后复用模拟数据 loader。

```bash
REAL=/data/v2x-radar-c
REAL_META=/data/metadata/v2xradar_supp_v1
REAL_RUNS=/data/runs/v2xradar_supp_v1
# 从官方 train 确定性抽取 4 个整 sequence 作开发；官方 validate 保留为最终 test 身份
python scripts/supplement.py prepare --dataset v2xradar --data-root "$REAL" \
  --manifest-dir "$REAL_META" --dev-count 4 --split-seed 17

python scripts/supplement.py plan --suite real_cooperative --data-root "$REAL" \
  --manifest-dir "$REAL_META" --output-root "$REAL_RUNS" --device cuda:0 \
  --final-model-file "$RUNS/frozen_final.json" --write-configs
RC="$REAL_RUNS/configs/supp_v2xradar_deer_sbr_S0_e30_s17.yaml"
python scripts/supplement.py preflight --config "$RC"
python scripts/supplement.py check-data --config "$RC" --samples 2
```

实际发布版本的 sequence 数/类别计数以准备输出为准，不硬写成 30 个。可用 `--dev-sequences` 代替随机固定留出。若有可核验的天气来源，通过 `--weather-map` 提供 sequence→官方自然天气标签的 JSON；不根据目录名猜天气。

上例不擅自认定晴天。DEER 的类型监督和真实 `MDD-CORR` 需要在准备阶段添加 `--clean-sequences`，填入你已核验的官方 train 序列（可含划出的开发序列，以供类型精度诊断）。重新准备时用新的 manifest 目录；不要覆盖冻结的 manifest。没有晴天证据时，S0 / L4DR / CoBEVT 的入口继续可用，MDD-CORR 与需要类型监督的 F 预检会给出具体阻塞。MDD-CORR 只训练晴天子集，额外正常参考分支单列；要严格匹配训练池，可在其余真实配置中同样启用 `data.clean_only: true` 并改 run_id。

```bash
# 晴天监督来源核验完成后
python scripts/supplement.py run-suite --suite real_cooperative --data-root "$REAL" \
  --manifest-dir "$REAL_META" --output-root "$REAL_RUNS" --device cuda:0 \
  --final-model-file "$RUNS/frozen_final.json" --execute --skip-completed
# MDD 参考监督暂不可用时，用同一方法槽的 L4DR 后备入口
python scripts/supplement.py run-suite --suite real_cooperative_l4dr --data-root "$REAL" \
  --manifest-dir "$REAL_META" --output-root "$REAL_RUNS" --device cuda:0 --execute
```

V2X-Radar-C 使用官方 LiDAR ROI（全双向协同范围，无模拟 camera FOV）。官方公开累加器的 BEV 与新增 true-3D extension 分别命名；开发 AP3D07 属于 extension，不能当作原论文官方数字。详见方案 8.1 和 `opencood/project/official/README.md`。

L4DR适配保留FAD（PointNet++ MSG）、前景focal目标、双向MME及由融合分支产生门控的MSGF。训练GT仅作为`train_targets.fad_boxes`进入损失，不进入模型推理；模型读取当前真实`raw_radar`并在FAD后体素化。FAD阈值0.1、0.2m忽略带、前景loss权重2来自[作者配置](https://github.com/ylwhxht/L4DR/blob/fcf2d7f0b7c632f9c23f75f720fe691be519d732/K-Radar-main-repo/configs/cfg_PP_L4DR.yml)。CUDA使用现有`pointnet2_batch`扩展；CPU是同运算功能检查。四通道、逐agent可变点数、共同检测头和GT协议适配会写入结果身份，不能把本项目结果写成原生论文复现。K-Radar候选使用35轮AdamW/cosine，batch=3、lr=0.001、weight decay=0.01、min lr=0.0001；其他平台先使用V2X-R协同候选并检查收敛。

K-Radar 需要官方 `sprdr_<radar_idx>.npy` 稀疏产品、OS2-64 LiDAR、标定、v2.0 revised labels、description 和官方 split lists，不需要自动下载/抽取原始 tesseract。目录可以独立存放：

```bash
KR=/data/kradar
KR_RADAR=/data/kradar_sparse_product
KR_LABEL=/data/KRadar_refined_label_by_UWIPL
KR_META=/data/metadata/kradar_supp_v1
KR_RUNS=/data/runs/kradar_supp_v1
python scripts/supplement.py prepare --dataset kradar --data-root "$KR" \
  --radar-root "$KR_RADAR" --label-root "$KR_LABEL" --manifest-dir "$KR_META" \
  --train-split /data/kradar_official_split/train.txt \
  --test-split /data/kradar_official_split/test.txt --dev-count 4 --split-seed 17
python scripts/supplement.py plan --suite real_weather --data-root "$KR" \
  --radar-root "$KR_RADAR" --label-root "$KR_LABEL" --manifest-dir "$KR_META" \
  --output-root "$KR_RUNS" --device cuda:0 --final-model-file "$RUNS/frozen_final.json" --write-configs
```

其原始目录是 `<KR>/<seq>/os2-64/os2-64_<lidar_idx>.pcd`、`info_calib/calib_radar_lidar.txt`、`description.txt`；Radar 是 `<KR_RADAR>/<seq>/sprdr_<radar_idx>.npy`，labels 是 `<KR_LABEL>/<seq>/<label_name>.txt`。header 决定同步索引，不能假设同文件名即同步。使用官方 v2.0 loader 的 ROI [0,-15,-2,72,15,7.6] 与 53° 方位角 GT 规则；统一 pillar 网格选 .375m 以满足网络 stride，不能误称这是原生 K-Radar 模型。输出两类 AP 和 macro mean，使用官方 revised AP41。官方最终 test 的帧保持原样；其 58 个 sequence 与官方 train 都重叠，不能把它写成独立场景测试。GT 不按当前模态点数删减，单车协同层被显式旁路。

### 6. 结果、统计与预算曲线

```bash
python scripts/supplement.py convergence --run-dirs "${DEV_RUNS[@]}" \
  --output "$RUNS/reports/convergence.json"
python scripts/supplement.py summarize --results "$RUNS/final_eval" \
  --baseline deer_sbr/S0 --output "$RUNS/reports/main_summary.json"
python scripts/supplement.py bootstrap --config "$C" --split test \
  --predictions-a "$RUNS/final_eval/S0_s17_snow.scores.predictions.jsonl" \
  --predictions-b "$RUNS/final_eval/AB_s17_snow.scores.predictions.jsonl" \
  --repetitions 1000 --output "$RUNS/reports/paired_sequence_ci.json"
```

`convergence` 生成 JSON/CSV/PNG，保留训练 loss/lr/更新数及独立配方和中途快照身份。`summarize` 接受实际 `*.scores.json` / `scores.json`，按数据/协议/transfer/天气/扰动分组，输出 mean、样本标准差（ddof=1）和逐 seed 配对差；V2X-R 的 Fog/Snow 均值先在每个 seed 内计算，再统计其标准差；种子不足时标记不完整，不补齐。不同rho/半径、训练配方及预算快照分别成组；跨方法/seed共同GT和样本集合必须一致，Fog/Snow必须来自同一选中权重。多份不同预算的 baseline 不自动择优混算。请先按前述 evaluate 命令为各方法/seed/天气生成 `final_eval`，Snow 和其他最终测试均需显式许可开关。

bootstrap 校验两组完整预测、GT 与条件一致，以 sequence 为单位成对重采样并重新调用对应 evaluator 计算 AP；不平均逐帧 AP，不替代训练种子方差。三个数据集和重新训练/zero-shot/fine-tuned 结果分别记录。所有新结果在实际执行前均为空，不沿用截图或合成检查中的已知答案作为论文成绩。

固定场景检查可用 `python scripts/supplement.py visualize --config "$C" --predictions "$RUNS/eval/AB_s17_normal.scores.predictions.jsonl" --output "$RUNS/visualizations/scenes.json"`。默认取归档中固定排序的 5 帧，或通过 `--sample-ids` 指定 1–5 帧，输出真实观测 BEV、固定 GT 和预测框；不重新评分。

### 7. Windows无真实数据检查

本地检查使用UTF-8。若Anaconda的NumPy MKL与PyTorch出现OpenMP #15冲突，可在该检查进程选择sequential MKL；该设置不用于论文效率测量：

```powershell
$env:PYTHONUTF8 = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python -m unittest tests.test_supplement tests.test_supplement_integrity tests.test_l4dr_supplement -v
python scripts/supplement.py smoke --output outputs/supplement_cpu_check.json
python scripts/reproduce.py results
```

此处smoke仅运行合成前后向与推理，不更新优化器。没有真实数据时，native数据读取、训练和最终评分仍需在已准备数据及扩展的环境验证。实际执行模块与缺数据skip详见[VALIDATION.md](VALIDATION.md)。
