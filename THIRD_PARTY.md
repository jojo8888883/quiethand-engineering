# 第三方资产与贡献边界

本包保存的是项目自己的组织/适配/诊断/展示代码及实验配方。没有把上游模型、数据集或其训练成果算成本项目原创，也没有打包它们的源仓库、视频、CAD 或权重。

| 组件 | 本项目的用途 | 原来源 |
| --- | --- | --- |
| Qwen3-VL | 视频动作语义、区域与实体判断 | [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) |
| HaWoR | 手网格和时序运动恢复 | [HaWoR](https://github.com/ThunderVVV/HaWoR) |
| SAM2.1 | 物体/手区域分割与跟踪 | [SAM2](https://github.com/facebookresearch/sam2) |
| FoundationPose | 给定 CAD/RGB-D 的物体位姿估计 | [FoundationPose](https://github.com/NVlabs/FoundationPose) |
| MANO / SMPL-X | 手参数化模型与重建支持 | [MANO](https://mano.is.tue.mpg.de/)、[SMPL-X](https://github.com/vchoutas/smplx) |
| ARCTIC | 原生几何与数据接口检查 | [ARCTIC](https://github.com/zc-alexfan/arctic) |
| TACO | 模型流水线的 RGB-D、物体资产和事后参考 | [TACO 官方数据说明](https://github.com/leolyliu/TACO-Instructions)；原获取记录在 `quiethand/taco_source.py`、`taco_contract.py` |

具体冻结版本在 `records/checkpoint_plan.json`；旧脚本中的绑定值用于重放旧实验，不代表推荐现在安装的最新版本。

## 发布状态

这份包是工程留档。未选择全包开源许可证，尚未公开发布。私有 GitHub 留档不自动授予开源许可；新增开源许可或公开数据样例由项目所有者另行决定。

用户手写评语、账户信息、数据访问凭证、原视频和模型文件不在导出选择中。原工作区记录仍保留，没有为了清爽而删除失败实验或用户意见。
