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
