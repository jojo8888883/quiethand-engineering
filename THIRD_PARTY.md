# 第三方资产与贡献边界

本仓库包含项目自身的组织、适配、诊断、展示代码及历史实验配方；不包含下列上游模型、数据集、权重、视频、CAD 或其训练成果。克隆本仓库不授予这些资源的访问权或使用许可。

| 组件 | 本项目中的用途 | 上游来源 |
| --- | --- | --- |
| Qwen3-VL | 视频动作语义、区域与实体判断 | [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) |
| HaWoR | 手网格和时序运动恢复 | [HaWoR](https://github.com/ThunderVVV/HaWoR) |
| SAM2.1 | 物体/手区域分割与跟踪 | [SAM2](https://github.com/facebookresearch/sam2) |
| FoundationPose | 给定 CAD/RGB-D 的物体位姿估计 | [FoundationPose](https://github.com/NVlabs/FoundationPose) |
| MANO / SMPL-X | 手参数化模型与重建支持 | [MANO](https://mano.is.tue.mpg.de/)、[SMPL-X](https://github.com/vchoutas/smplx) |
| ARCTIC | 原生几何与数据接口检查 | [ARCTIC](https://github.com/zc-alexfan/arctic) |
| TACO | RGB-D、物体资产和事后参考的历史接口 | [TACO 说明](https://github.com/leolyliu/TACO-Instructions)；相关代码为 [`quiethand/taco_source.py`](quiethand/taco_source.py) 与 [`quiethand/taco_contract.py`](quiethand/taco_contract.py) |

历史冻结版本记录在 [`records/checkpoint_plan.json`](records/checkpoint_plan.json)。旧脚本中的固定参数只用于说明既有实验环境，不建议替代上游文档或资源许可。

## 许可证状态

仓库已公开，但项目代码的许可证尚未选定。本文件不授予任何额外代码、数据或模型许可，也不改变上游组件各自的许可和访问要求。
