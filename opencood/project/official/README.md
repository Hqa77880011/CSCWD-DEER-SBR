# 官方评分源

- `v2xradar_eval.py` 原样取自 [V2X-Radar eval_utils.py](https://github.com/yanglei18/V2X-Radar/blob/1624751e31260c68d544e60599079d2104ee8a73/CodeBase/OpenCOOD/opencood/utils/eval_utils.py)，保留文件内署名及许可。调用其 `calculate_ap`，匹配使用相同的逐帧置信度排序、删除已匹配 GT 和全局 VOC2010 积分。官方公开 OpenCOOD 文件使用平面 polygon IoU，**其 AP 不冒称为 3D AP**。新增真实 3D IoU 的结果明确标为 `true3d_extension`，BEV 单列。ROI 来自官方 LiDAR 配置，未使用模拟数据的 camera FOV。
- `kradar_eval_revised.py` 取自 [K-Radar eval_revised.py](https://github.com/kaist-avelab/K-Radar/blob/f90c28993d1757ce0236632b4d912c92a14976b2/utils/kitti_eval/eval_revised.py)，保留官方匹配、41 点积分和修正后的 `z_center=0.5`。仅将文件顶部 GPU IoU 导入替换为同接口的 CPU 精确 polygon 后端；不改 AP 累加与插值。该后端遵循原 `nms_gpu.py` 的顺时针 yaw 约定。

两个源版本均于 2026-10-23 核对。K-Radar 标签解析和坐标转换对应同版本 `datasets/kradar_detection_v2_0.py` 及 `utils/util_pipeline.py`。K-Radar 官方 train/test 列表各包含全部 58 个 sequence；本实验的开发集从官方 train 按整 sequence 留出，官方最终 test 帧完整保留。因此 train/development 不共享 sequence，官方 test 与它们仍存在官方固有的 sequence 交叠，不能声称最终测试是未见场景评测。

K-Radar 另对少于 10 帧的诊断子集限制 `num_parts <= num_frames`，避免空分块的 concatenate 错误；不改变匹配或 AP 插值。V2X-Radar-C GT 复用与该官方版本一致的 `create_bbx → x1_to_x2 → corner_to_center → any-corner ROI` 链路，保留 roll/pitch 转框语义，区别于模拟 V2X-R 的专用 fixed-GT upright enclosure。
