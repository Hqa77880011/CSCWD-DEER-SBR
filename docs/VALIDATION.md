# 验证记录

验证日期：2026-10-21（Asia/Shanghai）。工作目录为独立导出的仓库，未覆盖原训练目录。

| 检查 | 结果 |
|---|---|
| Python 静态编译、Shell 语法检查 | 通过 |
| 全部 unittest（启用已准备的真实数据） | **155 项全部通过，无 skip** |
| GitHub 重新克隆 + 独立 Python 环境 | 2 项可移植入口回归和参考结果核算通过；不依赖原项目的 editable 安装 |
| Cython 及四个 CUDA 扩展重新编译 | 通过；GPU 架构 7.5 |
| GPU 环境检查 | 导入、spconv 前向/反向、IoU/NMS、points-in-boxes、合成检测训练与 checkpoint 均通过 |
| 固定数据版本在线探测 | Hugging Face 元数据下载及固定源校验通过 |
| 完整数据元数据重建 | 从 171 份已有归档校验回执重建模态/图像尺寸 manifest，原点云仅通过独立目录中的链接读取 |
| 固定 GT 重建 | 全部 12078 个样本，train/validate/test 计数、GT 计数及语义哈希与记录完全一致 |
| 真实 train 辅助监督审计 | 32 个固定样本通过；可信正样本覆盖、来源 ID 和 Fog 标签屏蔽条件通过 |
| AB 正式运行预检 | 数据路径存在，`formal_blockers=[]` |
| AB 真实 GPU smoke | 2 个 microstep、1 次有效优化更新；loss/梯度有限，源码未变化 |
| 七组参考结果表核算 | 7 行 Fog/Snow 均值和相对 S0 增益均通过 |
| 上传文件检查 | 无数据、训练权重、Python 环境、编译二进制或凭据模式匹配 |

核心 Python 源码 SHA256 与正式实验记录一致：

```text
71459fab18d8ad489c84cf3002bd283750b15444964f09f359b0772fcbf697e1
```

本次 GPU smoke 使用 FP32、microbatch 4、累积 2，峰值 allocated 约 12.78GiB，reserved 约 13.95GiB。该结果是一次有界功能验证，不是长期训练资源报告。

测试环境为 Python 3.10.8、PyTorch 2.1.2+cu121、CUDA 12.1、RTX 2080 Ti（本机显示约 22GB 显存）。观测到的依赖版本记录在 `repro/verified_environment.json`，可供重建环境的约束文件为 `requirements-repro-lock.txt`。

本次复用了现有 Python 训练环境，并在导出目录重新编译扩展、运行真实数据测试；没有重新安装整套 CUDA 依赖，没有重复下载约 161GB 数据，也没有重新训练七组 20 epoch 或重新跑 21 项完整测试。表格中的 AP 来自保留的原始正式实验记录，不能将本次 smoke 当作新的 AP 测量。GitHub Actions 模板尚未激活。

机器可读的检查摘要见 `repro/verification.json`。完整本地日志位于忽略上传的 `outputs/verification/`。

## v1.0.0 训练权重 Release 验证

本次扩展通过 **159 项回归测试，无 skip**。七份原始 selected checkpoint 全部通过 SHA256、内嵌配置、数据文件身份和精确 BN 统计加载校验；10 个 Release 附件的 GitHub 服务端 SHA256 和大小与本地清单一致。已用 README 的下载入口重新下载 AB 权重与共享元数据，并在重新克隆的仓库、新数据视图路径下完成加载；独立 Python 环境中的 4 项 Release 入口回归也通过。

直接评测入口实际重跑了 AB/Snow 的完整测试：**3166 样本、30296 GT，AP3D@0.7=49.19382268909672%**。三个 IoU 阈值的 AP3D 与 APBEV 共六项数值均与原正式测试完全相同，冻结 BN 未变。预测 JSONL 的 SHA256 也与原正式实验完全相同，即逐字节一致：

```text
dd13771fa8ca73b92b297fdc25375d2371a3d162b613433406d821fffb248de4
```

其余 20 项完整测试未重复执行；它们的原始权重、配置、数据和评测身份保持不变。机器可读记录见 `repro/pretrained_release_verification.json`，原并行导出流程的完整验证等价记录见 `repro/pretrained_parallel_equivalence.json`。


## 2026-10-23 补充实验实现核查与修复

本节为schema 5 / `supplement_v1`当前检查；前文2026-10-21为原七组历史记录。本轮没有下载数据或权重、运行下载脚本、连接实验服务器、正式训练或全量评测；没有生成新增论文AP、收敛或资源成绩。

本地环境：Windows、Python 3.13.5、PyTorch 2.7.1+cu118、NumPy 2.1.3。模型功能检查在CPU执行。18个相关测试模块实际执行**163项：160通过、3项skip、0 failure、0 error**，未运行全仓库测试。完整模块与skip身份记录于`results/supplement/local_regression.json`，详细本地日志位于忽略上传的`outputs/verification/supplement/related_regressions.txt`。

| 实际检查 | 结果与范围 |
|---|---|
| 三模板、方法注册、15个suite及旧CLI别名 | 配置展开检查通过；20/30/60/90独立配方、原生40轮及K-Radar L4DR 35轮分别记录；F仅使用标明为合成的临时开发证据 |
| 模型矩阵 | **21路CPU合成前向/反向/推理通过**，零优化更新；14个内部变体、5个外部方法、Where2comm+AB、K-Radar双类别；CoBEVT检查非整除窗口，MDD检查单agent/空Radar |
| L4DR缺失机制修复 | 6项专用回归通过：FAD前景/忽略带、FAD后体素上限、双向MME标量及偏移已知答案、MSGF门控输入、K-Radar格式完整网格的FAD loss/梯度及无标签推理、空Radar、原生35轮AdamW候选；与其他用例有合并验证关系 |
| F开发冻结与角色隔离 | 完整开发日志、按规则选择的epoch、匹配数据/损失/公共结构/源码、Normal容忍线、唯一赢家复算通过；online resume/selected/budget snapshot区分通过 |
| 统计/evaluator已知答案 | AP41、BEV/true-3D区分、重算AP的sequence bootstrap、逐seed配对、Fog/Snow同权重、天气均值标准差、rho分组、共同GT/样本身份检查通过 |
| 数据格式与几何 | V2X-Radar-C真实车辆ego/路侧配对、K-Radar同步header/标定/类别/单车样例、固定GT、投影与来源监督回归通过；输入均为合成格式文件 |
| 原七组与分发源码包 | 七份原配置字节与归档科学源码身份检查通过；Windows发布helper与源码包同步；`scripts/reproduce.py results`、归档CLI、七份重定位配置、默认路径/参数传递检查通过；未加载真实权重重新评测 |
| CLI/静态/文档 | 27次CLI调用通过；35个受影响Python文件语法编译通过；7份Markdown的81处本地链接通过，无旧方案路径；缺数据预检正确返回阻塞 |
| native环境预检 | **blocked**：缺cumm/spconv、open3d、pypcd、PointNet++ CUDA扩展及box_overlaps编译扩展；导入检查未替代CUDA数值检查 |
| 真实数据全链路 | 未运行V2X-R/V2X-Radar-C/K-Radar真实样本训练或最终评分；数据字段/单位/实际标定、训练充分性和方法精度仍需真实运行 |
| CUDA与效率 | 同步、跨帧释放和计时边界已检查；没有本轮CUDA耗时/显存实测，FAD CUDA采样/球查询/三邻域尚未验证 |

三个skip均属于`test_fixed_gt.FullPreparedDatasetTests`：全官方样本覆盖、全部GT来源与几何、全量天气不变性/空样本保留；原因都是本地无完整真实数据。合成样例中的已知答案不写入`results/reference/`或论文结果。

本轮先复现并修复的旧错误分别为：`prepare_data.normalize_manifests`的POSIX receipt键在Windows转换为反斜杠；`pretrained.unpack_metadata`在文件仍打开时替换；`pretrained.dataset_view`缺符号链接权限；以及扩大到相关固定GT检查后发现的camera/source路径键问题。前3项发布helper也同步到源码分发包，原七组科学源码、配置、参考结果和权重身份未放宽。固定GT主源码改用POSIX相对键，在Linux上保持相同语义，归档原科学源码保持原字节。

早期组合回归曾被**OpenMP #15**中断，该次不计为通过。定位到本机Anaconda MKL与PyTorch运行库重复加载后，最终检查进程使用`MKL_THREADING_LAYER=SEQUENTIAL`、`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`及`PYTHONUTF8=1`完成；未使用`KMP_DUPLICATE_LIB_OK`。此设置仅用于本地功能检查，不用于正式性能结论。

修复内容还包括：F冻结规则和checkpoint证据的复算；统计拒绝不同GT/场景或天气混权重；参数敏感性独立分组；原生/独立配方run_id隔离；实际通信距离的agent数量过滤；晴天开发集的正常类型标签与未知天气屏蔽；供体RMS/余弦与raw/tanh诊断；L4DR的FAD/MME/MSGF与额外前景目标。项目主README与上游readme移为不同路径，避免Windows大小写覆盖。

正式方案为[13_补充实验方案与实施说明.md](13_补充实验方案与实施说明.md)。可核查摘要还包括`results/supplement/local_smoke.json`、`local_environment.json`和`local_cli_checks.json`。HRCP尚未取得足够可核查实现资料，REL作者仓库仍仅代码待发布说明；二者未伪造训练配置，外部限制不阻塞已实现矩阵。

## 2026-10-23 服务器补充实验命令整理

完整服务器步骤统一维护在 [REPRODUCING.md 的补充实验入口](REPRODUCING.md#补充实验复现入口schema-5)，README 同步导航。本轮只修改文档，没有连接实验服务器、下载数据或权重、启动训练或推理评测；前述模型回归结果没有重新运行或改写。

| 本次实际检查 | 结果 |
|---|---|
| Bash 语法 | 22 个 Bash 代码块通过 Git Bash `bash -n`；只解析，未执行其中命令 |
| 内嵌 Python 语法 | 5 段 heredoc / `python -c` 编译通过 |
| CLI 参数 | 54 个补充实验命令模板展开为 72 种参数情况，调用实际 CLI parser 验证通过；未进入 action 执行逻辑 |
| 不依赖 F 的配置展开 | 9 个 suite 在内存展开并校验通过；内部主矩阵 18 项、外部主矩阵 12 项、核心机制 9 项、扩展机制 15 项；未写配置或访问数据 |
| F 依赖 | rho、radius、second_backbone、real_cooperative、real_weather 只检查参数和冻结文件参数绑定；没有伪造冻结记录，没有执行其配置生成或训练 |
| 预检门槛已知答案 | 无阻塞时放行；缺输入路径或有 `formal_blockers` 时拒绝，3 种内存样例通过 |
| 最终评测任务读取 | 从 12 项原生外部方法清单正确读取配置、数据集和对应 selected 路径；未加载权重 |
| Markdown 与差异 | 3 份文档的 40 处本地链接/锚点、Git 路径大小写、代码围栏和 `git diff --check` 通过 |

命令区分原生方法轮数与受控预算、先导复用与 online 恢复、开发冻结与最终测试；鲁棒性按 suite 分别汇总，避免重复 baseline 被当成同一 seed 的多条记录。真实数据准备、CUDA 算子/新模型检查、实际数据读取、正式训练和全量最终评分均待服务器执行。所有本轮临时检查均在内存进行，没有新增实验结果或临时产物。

## 2026-10-23 补充实验执行性能优化

本轮实际修改训练/验证调用链和配置生成入口，未调用多代理、连接实验服务器、下载数据或权重、启动正式训练、读取真实数据评测或执行 CUDA benchmark。环境仍为 Windows、Python 3.13.5、PyTorch 2.7.1+cu118；检查进程显式 `CUDA_VISIBLE_DEVICES=''`，使用前述 sequential MKL / 2 CPU threads 设置。

12 个相关模块共 **112 项：109 通过、3 项因完整真实数据缺失 skip**。模块为 `test_supplement_performance`、`test_bn_runner`、`test_batchnorm_refit`、`test_training_protocol`、`test_supplement`、`test_supplement_integrity`、`test_validation_fastpath`、`test_v3_mechanisms`、`test_supported_residual_losses`、`test_data_geometry_review`、`test_v4_protocol`、`test_fixed_gt`。日志为 `outputs/verification/supplement/performance_regressions.txt`；新增定向回归见 [test_supplement_performance.py](../tests/test_supplement_performance.py)。后续配置绑定/完整辅助 loss 检查增强仅重跑受影响用例，也通过。

| 本次实际检查 | 数值标准与结果 |
|---|---|
| 梯度有限性合并 | 正常路径不改任何梯度；注入 NaN、Inf 精确返回对应参数名，None 梯度合法。真实 runner 的合成 NaN loss/梯度在 optimizer 更新和 online checkpoint 保存前抛错 |
| loss 标量批量读取 | 对 FP32 loss 与大整数计数一次转 CPU，以 float64/Python double 求均值；与原逐标量 Python 求和结果严格相等，`rtol=atol=0`，无额外 autograd 路径 |
| activation checkpoint 开/关 | 同初始权重、32×32 合成网格、真实 AB 模型与 SupplementalLoss，包含 detection/type/correction/identity loss；`psm`、`rm`、总 loss/各项 loss、全部存在的参数梯度和 BN buffers **逐位一致**（`rtol=atol=0`）。BN 不因局部重计算多更新 |
| runner 的普通/同步计时与恢复 | CPU Linear+BN+Dropout 合成模型，microbatch4/累积2，4 次有效更新；两种计时模式以及 2+2 中断续跑的模型、Adam、scheduler、sampler、TrainState、Python/NumPy/Torch RNG **逐位一致**。loss 与日志项一致；测试替换了数据/模型工厂，未执行真实数据 benchmark |
| checkpoint IO | 合成 runner 保存间隔2，实际仅在第2/4次更新提交，末尾同状态不重复写；online role 与 BN role 保持。既有 selected/budget snapshot 禁止 resume 及 checkpoint 状态回归通过；原子保存/fsync 未删除 |
| raw cache / 数据 | K-Radar 合成格式文件，在 epoch 0→1→0 比较关闭/开启缓存：点云体素、训练标签、增强/扰动、来源 metadata 和固定 GT 严格相同；原始数组未被坐标变换污染；文件变化重新读取、缓存字节受限。既有 V2X-R PCD/YAML 哈希篡改拦截回归通过 |
| loader / 验证复用 | 0/2 workers、prefetch1/2 的两轮样本顺序严格相同，pin_memory/persistent_workers/prefetch 实际属性核对通过。两次验证只构建一次各天气输入流，但每次重新调用预测；普通模式无同步 profiler callback。BN 输入流复用后仍重新拟合并恢复原 BN/RNG |
| 保留的高效实现 | `efficient_features` 的既有路由比较为严格相等；受支撑残差比较 `rtol=2e-5, atol=1e-7`，稀疏辅助 loss/梯度比较 `rtol=2e-6, atol=1e-7`，均通过。后两者因 FP32 运算/归约顺序不同沿用原测试容差，本次未放宽 |
| 科学协议与旧身份 | FP32、microbatch4/累积2/有效batch8、采样/天气/固定 GT、每轮全量开发验证、选模容差和 BN 政策未改。原七组配置与归档科学源码身份回归通过；`repro/`、`results/reference/` 和原 schema4 配置无差异 |
| 候选与 schema | 7 个阶段共14种候选在内存展开并校验；只改变资源、activation checkpoint、online 保存间隔及输出身份。RAM 校验覆盖所有同时存活的 loader cache，非法 workers/prefetch/线程值被拒绝；未放开 FP32 schema |
| CLI 与文档 | 实际生成并读回2份 workers YAML；向 pilot/pilot_sup/internal_main/external_main 传递执行设置，确认原生 batch/optimizer/schedule、数据/损失/选择规则不变。11组实际 parser 检查、12个受影响 Python 文件内存编译、31个 Bash 块 `bash -n`、两份文档本地文件链接及 `git diff --check` 通过 |

容差依据：执行调度、缓存和日志不改变 CPU FP32 运算，因此新等价检查采用严格相等；已有稀疏特征路径采用其原有 FP32 容差。这些合成结果**不等于真实数据训练轨迹等价，也不证明 GPU 加速或显存收益**。正式效率入口仍前后同步 CUDA，并显式关闭 raw array cache以保留每帧文件读取/解析；该入口本轮未在 CUDA 上执行。

初次组合测试由 `python -` 启动，Windows spawn 无法重新加载 `<stdin>`，造成一项 DataLoader 启动错误；改为下面的模块入口后整组通过，没有修改采样逻辑来规避该错误。三个最终 skip 均为原 `test_fixed_gt.FullPreparedDatasetTests` 的全量数据检查。

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) 'tests')
$env:CUDA_VISIBLE_DEVICES = ''
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '1'
python -m unittest test_supplement_performance test_bn_runner test_batchnorm_refit test_training_protocol test_supplement test_supplement_integrity test_validation_fastpath test_v3_mechanisms test_supported_residual_losses test_data_geometry_review test_v4_protocol test_fixed_gt -v
```

性能证据仅有旧并行验证摘要，没有补充实验原始训练耗时；未产生新加速比、训练收敛或 AP。候选生成、有界测量、正常/诊断计时区别、资源总量及选择标准统一见 [REPRODUCING.md](REPRODUCING.md#11-性能证据与计时口径)。本地 CLI 生成的临时候选在核对后移入回收站；仅保留上述检查日志，不新增平行方案或实验结果文件。

## 2026-10-24 README 分阶段服务器训练命令

将已有复现指南中的各阶段命令直接补入 [README 的服务器训练入口](../README.md#server-training-commands)，覆盖环境/执行配置、内部与外部先导、独立预算、三种子、F冻结、核心与扩展机制、单因素、第二骨干、两套真实数据和online恢复。本轮只修改文档，未连接实验服务器、访问真实数据、下载或启动训练/评测。

实际检查：17个新增Bash块通过 `bash -n`；49组项目命令参数调用实际parser校验、4段内嵌Python编译通过；7个不依赖F的suite及8种独立预算展开检查通过，外部原生30/40轮身份保持；5处新增本地文件/锚点链接与 `git diff --check` 通过。F依赖阶段只验证参数与冻结文件传递，未伪造F记录或生成对应配置；未因文档修改重跑模型测试。全部检查在内存进行，没有新增临时产物。

## 2026-10-24 配置接线、方法依据与训练开销核查

本节为当前代码结论。原七组配置、参考结果、Release身份和legacy源码包保持原样；前文日期更早的检查描述其当时版本，不能用来推断本轮L4DR修复前后训练轨迹相同。本轮未调用多代理、连接实验服务器、下载数据/权重、启动正式训练或真实数据评测，也未执行CUDA benchmark。

本地环境为Windows、Python3.13.5、PyTorch2.7.1+cu118、NumPy2.1.3。核验环境时发现PowerShell赋值空字符串会移除 `CUDA_VISIBLE_DEVICES`，不能据此判断CUDA已隐藏；当前CPU检查改用 `-1` 并实际确认 `torch.cuda.is_available()==False`。模型矩阵始终显式在CPU执行，未执行CUDA模型计算或benchmark。

原文与作者源文件的采用版本、逐方法A/B/C/D结论见[方案§8.2](13_补充实验方案与实施说明.md#82-原文作者版本与当前适配的核查结论)；有效参数与资源单位见[复现指南§0.1](REPRODUCING.md#01-有效配置参数范围与恢复身份)。本轮先复现了错误hypes/variant、无效optimizer/schedule/BN配置被接受、外部方法携带内部开关、BN逐模块传输和L4DR分数未detach等触发条件，再修复并检查相应路径。

| 实际检查 | 结果、比较对象与数值标准 |
|---|---|
| 定向回归 | **10个模块、105项全部通过，0 skip/0 failure/0 error**。模块为 `test_supplement_performance`、`test_supplement_integrity`、`test_l4dr_supplement`、`test_supplement`、`test_bn_runner`、`test_batchnorm_refit`、`test_training_protocol`、`test_validation_fastpath`、`test_v4_protocol`、`test_data_geometry_review`。未运行全仓库测试 |
| 当前配置的模型矩阵 | **21路CPU合成前向/各项loss/反向/推理通过，零优化更新，无AP**。14内部变体、5外部方法、Where2comm+AB、K-Radar双类别。smoke现在使用实际配置的loss权重、efficient_features和输入/target筛选，避免合成入口硬编码另一条loss路径 |
| 配置真实生效 | CLI省略预算得到60/30/30；15份pilot/external_main配置和5份单项配置写入临时目录并读回；执行设置不覆盖原生batch/optimizer/schedule/BN，外部无局部模块时activation/efficient/内部loss为false/false/0。num_workers、prefetch、persistent、pin的实际loader属性及0/2 worker样本顺序检查通过。独立micro4/累积2/有效8、原生无累积及类别选择标签均不能以手改配置绕过 |
| Where2comm执行优化 | 在32×32合成网格，将旧的无用count/.item在原位置恢复为reference hook，再与删除后的前向及筛选输入比较；psm/rm/通信比例、loss、所有存在的梯度、BN buffers及Python/NumPy/Torch RNG **逐位相同，rtol=atol=0**。原有两次backbone/shrink调用及其BN更新未删 |
| activation checkpoint | 开/关同初始AB真实模型、完整辅助loss；输出、各loss、梯度和BN buffers严格相同，`rtol=atol=0`。仅无BN局部分支重算；无局部模块外部方法拒绝开启该开关 |
| BN快照与有限性 | 三个BN的CPU传输调用从6次降为2批，统计/模式/参数/RNG不变；int64的 `2^40+1` 更新数精确保留，NaN及负variance仍识别。NaN loss/Inf或NaN梯度仍在optimizer更新/online保存前拦截 |
| L4DR目标与梯度 | FAD→voxel分数detach后，单独检测loss不产生FAD梯度，MME仍有梯度；加回前景focal后FAD梯度非零且有限。前景/外扩忽略带、首点voxel cap、双向MME偏移、MSGF来源、空Radar及无GT推理回归通过。该detach更正不以旧训练梯度等价为验收标准 |
| L4DR插值已知答案 | 三邻点距离为[0,2e-8,4e-8]时，作者加法epsilon的权重比为15:5:3；旧clamp为4:2:1。实际FAD传播输入与解析值比较采用 `rtol=2e-6, atol=1e-7`，容差仅覆盖FP32倒数/归约与有理式运算顺序。内存恢复旧clamp后该回归确实失败；未放宽容差 |
| MDD构造器与参考loss | 有效hypes的beta端点对应实际三个buffer；与作者 `make_beta_schedule` 计算逐位一致。epoch0/19/29的参考loss和预测梯度，与保留的原生 `PointPillarLossMDD` 函数严格相同；参考target不获梯度，`rtol=atol=0`。自然未知天气参考拒绝回归通过；没有声称Normal参考与作者noise-masked参考等价 |
| optimizer/scheduler与恢复 | K-Radar L4DR按作者eps=1e-8、CosineAnnealingLR的一轮T_max逐更新执行；12次合成更新的实际LR与原生PyTorch scheduler逐值相同。完整12次与4+8恢复的模型、AdamW、scheduler、sampler、TrainState和RNG严格相同。内部4次与2+2恢复、正常/同步计时模式同样严格相同；均使用替换的数据/模型工厂，无真实数据 |
| 数据/缓存/固定GT | 合成K-Radar格式文件的cache开/关、epoch0→1→0，点云/标签、样本、天气、扰动、来源metadata和固定GT严格相同；LRU字节受限、文件变化重读。V2X-R PCD/YAML哈希篡改拦截及几何/FOV回归通过，未缓存增强后的结果或模型特征 |
| 身份与角色 | `--skip-completed`拒绝源码或GT元数据变化、错误selected/resume角色；online、selected、budget snapshot恢复边界通过。完整配置/源码/数据恢复检查保留。原七份配置与归档科学源码的字节/身份回归通过；未读取Release真实权重重新评测 |
| CLI、静态与文档 | 8次实际无数据CLI action通过；12个修改Python文件内存编译通过；58个Bash块、75处本地文件/锚点通过检查，Bash仅做 `bash -n`，未执行服务器命令。检查发现并修正方案中一处过期数据准备锚点；`git diff --check`通过 |

最终回归日志为忽略上传的 `outputs/verification/supplement/config_method_review_final.txt`，CPU矩阵为 `config_method_smoke.json`，CLI/静态检查摘要为 `config_method_cli_checks.json`（均在同目录）。临时候选、夹具和检查脚本使用回收站清理，不作为新实验结果保留。

可重跑的定向检查（Windows、本地CPU，不读取真实数据）：

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) 'tests')
python -m unittest test_supplement_performance test_supplement_integrity test_l4dr_supplement test_supplement test_bn_runner test_batchnorm_refit test_training_protocol test_validation_fastpath test_v4_protocol test_data_geometry_review -v
python scripts/supplement.py smoke
```

**真实未决项。** CPU检查没有验证L4DR CUDA FPS/ball-query/three_nn与CPU的数值对应、spconv真实预处理、真实字段/标定/训练轨迹或AP；已安装CUDA扩展二进制身份也须在服务器确认。CoBEVT/Where2comm的project epoch warmup/cosine、MDD参考目标/beta、Where2comm单轮机制和真实数据适配仍按方案§8.2登记，未改成新的科学协议。没有本轮正常吞吐、强制同步诊断、正式推理延迟或显存前后对照，所有收益待测，不填写加速比。

已准备数据与CUDA环境的服务器上，最小下一步是为本轮源码生成新配置，先 `preflight` / `check-data --samples 2`，再对L4DR等受影响方法运行 `project_tools/run_experiments.py --config "$C" --mode smoke --warmup 0 --steps 1`；该命令会读取真实train并执行一次更新，只供服务器后续验证。随后使用[有界测量命令](REPRODUCING.md#12-分阶段候选配置与有界服务器测量)分别运行throughput和synchronized诊断；逐epoch完整验证、BN政策及选模不变。这些服务器步骤本轮均未执行。
