# teacher_free_v3：BN修复前的精简可复现归档

冻结于2026-10-19。归档保留原v3模块、损失、FP32 / microbatch4 / accumulation2 / workers8 / 轻分支重计算，以及尚未解决的BN运行统计失配。它不是修复后的训练协议，不含正式训练或AP结果。

`teacher_free_v3_source.tar.gz` 包含440个文件、19份当前v3配置与78项pending矩阵，压缩大小768,106字节。源码指纹严格等于原验收的 `c02a351174bd9736e5e7240d4ebe6df9e6ff6093a1a0d1691a4166dd1817105c`；归档SHA256为 `31d509c5696eeab1d9ecdbb4f0499a0e331b5356887add76c4de6f0cf7af92ec`。逐文件大小和SHA256见 `manifest.json`，全部成员重读校验及源码指纹结果见 `verification.json`。

包内有 `V2X-R_project/opencood`、`project_tools`、`tests` 的源码与扩展构建源、当前配置/底座配置、docs、环境依赖清单、关键验收/资源/学习/恢复报告及BN失败证据。排除项目数据、环境、输出权重、Git、build、已编译扩展和Python缓存；`opencood/data_utils/datasets/` 是必要的loader源码，已保留，不属于被排除的原始数据目录。v2归档未改动。

原验收为114项回归通过，v3配对吞吐6.3456协同样本/s、峰值allocated/reserved为12.2775/18.2285GiB。25更新诊断权重在train[0:8] Normal的eval loss约2.08e6、batch-stat反事实约3.457，失配约60万倍；v2同样失败。batch-stat结果只用于诊断，不能作为推理策略、选模分数或AP替代。

在独立目录复现，避免覆盖当前工作树：

```bash
CSCWD_V3_REPLAY=/tmp/cscwd_teacher_free_v3_replay
mkdir -p "$CSCWD_V3_REPLAY"
tar -xzf /root/autodl-tmp/CSCWD/setup/teacher_free_v3_archive/teacher_free_v3_source.tar.gz -C "$CSCWD_V3_REPLAY"
source /root/autodl-tmp/CSCWD/setup/activate.sh
cd "$CSCWD_V3_REPLAY/V2X-R_project"
export PYTHONPATH="$PWD"
bash project_tools/build_extensions.sh
python -m unittest discover -s tests -v
python project_tools/run_experiments.py --config cscwd_configs/project_v3_W_AB.yaml
```

最后一行只预检。环境仍使用 `/root/autodl-tmp/CSCWD/envs/cscwd`；ABI相关扩展须在副本中重建，或明确复用同一环境已核验扩展。配置仍引用现有只读数据、固定GT/划分/来源审计。诊断运行前把 `artifacts.output_root` 和 `experiment.run_id` 改为独立路径，并记录变更哈希；不得覆写现有验收或将旧checkpoint接入BN修复后的训练协议。

以下不同状态的原始权重/恢复工件保留在原位置，包内不复制：

- `outputs/v3_paired_v3/20261019_132107_fp32_m4_w8/benchmark/latest.rank0.pt`：v3 25更新资源/BN诊断权重。
- `outputs/v3_paired_v2/20261019_130601_fp32_m4_w8/benchmark/latest.rank0.pt`：配对v2诊断权重，必须用v2归档加载。
- 当前跨epoch恢复的具体checkpoint路径见包内 `setup/teacher_free_v3_resume.json`；初始化及其他唯一状态的保留依据见 `setup/teacher_free_v3_cleanup_manifest.json`。

原始内存快照、数据、环境、唯一权重及旧v2/教师归档继续保留在工作区原路径。归档内文档记录冻结时状态；BN修复后的实际协议、诊断范围与验收以工作树docs/02、04、08为准。
