# 相近工作：以后做选择时的参照

来源：本项目 2026-09-17 对官方论文、项目页和代码入口的核对。本页只收录该次调研，不代表 9 月 19 日重新检索了全部最新进展；没有同数据、同预算性能复现。

| 工作 | 已覆盖的部分 | 与 QuietHand 重合 | 以后复用时值得看什么 |
| --- | --- | --- | --- |
| [NVIDIA Video to Data](https://github.com/nvidia-isaac/video_to_data) | 视频切段/实体关系、手物重建、机器人重定向与训练分支 | 基础处理流水线最直接 | 优先看 [ego 重建入口](https://nvidia-isaac.github.io/video_to_data/reconstruction/)；哪些组件可直接替换自己维护的代码 |
| [Open-AoE](https://github.com/ant-research/Open-AoE) / [论文](https://arxiv.org/abs/2607.14183) | 第一视角采集、动作标注、双手/相机轨迹、人工纠正、训练格式 | 从视频加工到可读取的数据 | 数据组织、检查界面、实际训练输入与标定要求 |
| [DeMiAn](https://arxiv.org/abs/2605.17077) / [代码](https://github.com/pearls-lab/demian) | 动作、场景、手臂姿态和推理四类文字用于 VLA/WAM | 更细语义标签帮助学习 | 它具体喂什么文字、哪些仿真任务有收益；不是所有模型都消费同样标签 |
| [EgoLoc](https://github.com/IRMVLab/EgoLoc) / [扩展论文](https://arxiv.org/abs/2508.12349) | 手部运动选帧 + VLM 定位接触/分离时间 | 几何辅助交互时刻标注 | 接触时刻如何接进下游；README 已有双手扩展链接，不能简单拿“双手”当空白 |
| [AGILE](https://agile-hoi.github.io/) / [v4 论文](https://arxiv.org/html/2602.04672v4) | VLM 引导物体建模、手物重建、接触/防穿透优化 | Agent + 几何 + 物理约束纠错 | 最新论文已有刚体双手评测；核对时核心代码尚未公开，不算现成可跑基线 |

## 具体判断

“把 VLM、手重建、SAM 和 FoundationPose 接起来”以及“标签更细”本身不足以成为独立创新。这个判断不等于这些项目在我们的每条视频上都更准。

V2D 关联的 [CHORD](https://arxiv.org/html/2607.00033v2) 已有仿真/真机实验，但接触力矩能力来自几何/模型指导，不是普通视频直接测得的实际受力；不同数据条件和实验分支不能混成“任何 RGB 视频都能直接教机器人”。

未来需要某一能力时，先比较输入条件、输出格式、可运行代码和已有失败案例，再决定采用上游还是复用 QuietHand 的局部模块。这里没有预选新的科研方向。
