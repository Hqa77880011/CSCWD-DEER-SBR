# 本次整理的验证记录

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
