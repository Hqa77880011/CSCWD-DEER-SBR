# Historical configuration templates

These files preserve the original implementation and registration matrix, including old absolute paths and 30-epoch plans. They are retained for protocol regression and source provenance.

Use `python scripts/reproduce.py configure` and the generated `outputs/reproduction/configs/*.yaml` for the **executed seven-variant, 20-epoch, seed-17** reproduction. The exact executed configuration snapshots are in `repro/configs/`. See [the reproduction guide](../docs/REPRODUCING.md). The registration matrix records planned work; it is not a completion record.

## 2026-10-23 补充实验

`supplement/` 的三个 schema 5 模板与 `matrix.json` 用于新增实验，不替换原七组配置。使用 `python scripts/supplement.py plan --suite pilot --output-root <新目录> --write-configs` 生成独立完整配方。当前方案与实现状态见[补充实验方案](../docs/13_补充实验方案与实施说明.md)，可复制命令见[复现指南](../docs/REPRODUCING.md)；新增训练和真实数据成绩均待运行。
