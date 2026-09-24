# 脚本索引

Python 可导入的核心包在 [quiethand/](../../quiethand/)，最小示例在 [examples/](../../examples/)。本目录同时保留工具函数和历史流水线脚本，不是统一的命令行 SDK。

## 可直接启动的本地服务

[serve_review.py](serve_review.py) 提供支持视频拖动的 HTTP byte-range 服务。输入目录、监听地址和端口均通过参数指定：

```bash
python3 scripts/quiethand/serve_review.py --directory /path/to/your/preview --bind 127.0.0.1 --port 8768
```

目录需要包含你自己的页面与媒体文件；仓库不带旧视频或点云。该命令不会自动打开浏览器。

## 可复用的几何函数

安装 `.[cpu]` 后，可从源码目录导入以下数值函数。调用方提供数组、相机参数和坐标变换，不依赖原服务器目录；各文件的批处理 `main` 不属于这个可移植接口。

| 文件 | 主要函数 | 用途 |
| --- | --- | --- |
| [diagnose_trajectories.py](m3_5_fusion/diagnose_trajectories.py) | `transform_points`、`trajectory_errors`、`relative_positions` | 坐标变换、轨迹误差与相对位置 |
| [mask_pose_consistency.py](m3_5_fusion/mask_pose_consistency.py) | `projected_mask_iou` | mesh 投影与区域一致性 |
| [build_fusion_preview.py](m3_5_fusion/build_fusion_preview.py) | `visible_surface_points`、`relative_envelope`、`pair_evidence` | 深度反投影与手物相对关系 |

函数签名见源码，输入示例见 [tests/](../../tests/)；含义与局限见[模块地图](../../docs/MODULES.md)。融合函数仍使用原实验的采样、深度尺度与阈值约定，接入前需逐项核对，不是适配任意数据的默认配置。这些源码工具不随核心 wheel 安装，需要保留仓库目录。

## 历史配方：供接入时参考

| 目录或入口 | 保存内容 |
| --- | --- |
| [m3_v1_2/](m3_v1_2/) | 完整动作区间、抽帧、语义、手与物体感知、中文结果页面 |
| [m3_jobs/](m3_jobs/) | 模型 runner 与公共运行工具 |
| [m3_5_fusion/](m3_5_fusion/) 的批处理脚本 | 坐标对齐、身份修正、案例诊断与前后对照 |
| [m3_eval_router/](m3_eval_router/) | 固定 30 条样本、6 条查看预算的质检比较 |
| `run_m0_*`、`run_m1_*`、`run_m2_*`、`run_m3_*` | 各阶段的历史运行和数据落地过程 |
| [setup_m3_remote_envs.sh](setup_m3_remote_envs.sh)、`prepare_*job*.py` | 原服务器环境与任务生成配方 |

历史脚本保留原实验 ID、路径和输入契约，用于理解既有结果；这些路径不属于本仓库提供的运行环境。接入新数据时先按[运行说明](../../docs/REUSE.md)准备真实输入与模型资源，再适配所需入口，不要直接执行原服务器安装或排卡脚本。
