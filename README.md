# QuietHand

第一视角手物交互的语义与几何对齐、诊断和可视化工具。

[项目页与演示视频](https://quiethand-brief-yuzhou.scut-lang.chatgpt.site/) · [快速使用](docs/REUSE.md) · [模块地图](docs/MODULES.md) · [结果与经验](docs/RESULTS_AND_LESSONS.md)

## 它做什么

QuietHand 将视频中的动作描述、手部运动和物体状态整理到同一套记录中，帮助查看模型输出、定位冲突，并保留来源和无法确定的信息。

```text
RGB 视频 ──→ Qwen3-VL ──→ 完整动作区间、动作与左右手语义
                           │
                           └──→ 按同一组源帧索引取帧
RGB 帧 ──→ HaWoR ──→ 手网格与轨迹 ──────────────┐
RGB + 物体提示 ──→ SAM2 ──→ 物体区域            │
物体区域 + 传感器深度 + 相机内参 + CAD           ├─→ 对齐与实体绑定
                    └──→ FoundationPose ──→ 位姿┘        │
                                                       ↓
                                结构化记录、冲突诊断、网页查看与质检排序
```

这个仓库提供三类工程资产：

- **结构化接口**：记录时间、坐标系、单位、来源和缺项；将语义区域绑定到明确的物体身份。
- **诊断与展示工具**：检查手物相对运动、投影与 mask 的一致性，展示结果并收集自然语言意见。
- **历史运行配方**：保留模型接入、数据对齐和固定样本实验的实现，方便理解与迁入其他项目。

## 快速开始

Python **3.10 或更新版本**。下面的接口示例只使用标准库，不需要 GPU、模型权重或数据集：

```bash
git clone https://github.com/jojo8888883/quiethand-engineering.git
cd quiethand-engineering
python3 examples/reuse_demo.py
python3 -m unittest discover -s tests -p 'test_reuse_demo.py' -v
```

示例使用明确标注的**人工构造输入**，展示物体身份绑定、未知信息、证据序列化和接触约束计算；不是视频推理结果。

安装核心包、运行全部 CPU 测试或接入自己的输入，见[运行与复用说明](docs/REUSE.md)。

## 从哪里复用

| 内容 | 入口 | 使用方式 |
| --- | --- | --- |
| 证据字段与物体身份 | [quiethand/](quiethand/) | Python API；接口与限制见[模块地图](docs/MODULES.md) |
| 最小接口示例 | [examples/reuse_demo.py](examples/reuse_demo.py) | 克隆后直接运行 |
| 几何诊断、页面与模型配方 | [脚本索引](scripts/quiethand/README.md) | 区分可调用函数、本地服务和历史批处理入口 |
| 已有结果与失败经验 | [docs/RESULTS_AND_LESSONS.md](docs/RESULTS_AND_LESSONS.md) | 小型结果记录，不含原始媒体 |

## 结果与适用范围

现有工程已接通过 Qwen3-VL、HaWoR、SAM2 和 FoundationPose。在一次固定的 30 条样本质检中，同样查看 6 条记录，加入现有几何信号选出了 2 条代理错误，语义基线选出 1 条。基线分数全部并列，目标来自单个 agent 的检查，因此这只是该样本内的观察；完整记录见[结果与经验](docs/RESULTS_AND_LESSONS.md)。

模型全链路需要 RGB-D、相机参数、物体 CAD 和各自运行环境，历史脚本并非任意 RGB 视频的一键标注产品。项目未验证标注降本、下游训练收益或机器人控制效果；理想接触约束的数学原型也尚未可靠接通真实视频输入。

## 文档与许可

- [模块地图](docs/MODULES.md) · [运行说明](docs/REUSE.md) · [验证记录](docs/VALIDATION.md)
- [相近工作对照](docs/RELATED_WORK.md) · [第三方组件与资产](THIRD_PARTY.md)

仓库不包含原视频、深度、CAD、模型权重或用户原始评语。代码许可证尚未选定，当前公开源码不附带开放源代码许可；上游模型与数据按各自条款获取和使用。
