# QuietHand · 阶段性工程资产

把第一视角操作视频里的动作语义、手部运动和物体状态整理到一起，并提供检查、纠错与结果查看工具。

这是 **2026-09-19 整理的工程原型**。适合以后接新项目时复用代码与经验；不是一个已经验证降本或机器人训练收益的方法，也不是任意视频的一键标注产品。

## 先看懂它在干什么

```text
操作视频的 RGB ──→ Qwen3-VL ──→ 动作区间、动作描述、左右手语义
                         │
                         └──→ 在动作区间中取帧
RGB 帧 ──→ HaWoR ──→ 手网格/轨迹 ──────────────┐
RGB + 物体提示 ──→ SAM2 ──→ 物体区域           │
物体区域 + 传感器深度 + 内参 + CAD              ├─→ 对齐、实体绑定
                    └──→ FoundationPose ──→ 位姿┘       │
                                                      ↓
                               结构化记录、冲突/缺项、网页查看、质检排序
```

已跑通的模型主线依赖 RGB-D、相机参数和物体 CAD，不是仅凭一段普通 RGB 视频恢复一切。几何没有自动把所有错误文字改对。末端也没有连接机器人控制器。

## 这次留下什么

| 想做什么 | 从哪里开始 | 可复用程度 |
| --- | --- | --- |
| 给手、物体、时间、来源统一字段 | `quiethand/models.py`、`schema/`、`m3_adapter_contract.py` | 可直接调用，但要按对应字段契约接入 |
| 防止“tool/target 名字”把物体 CAD 配反 | `quiethand/object_identity.py` | 可直接调用；当前是两个不同物体的契约 |
| 比较手物相对运动、检查投影与 mask | `scripts/quiethand/m3_5_fusion/` 中的几何函数 | 数值函数可复用；批处理入口依赖旧数据布局 |
| 接 Qwen、HaWoR、SAM2、FoundationPose | `scripts/quiethand/m3_v1_2/`、`m3_eval_router/` | 原运行配方留档，需自己的数据/模型/运行环境 |
| 把结果展示出来并收集意见 | `build_result_preview.py`、`review.js`、`serve_review.py` | 页面生成器和视频 Range 服务可复用 |
| 同预算比较“优先查看哪些片段” | `m3_eval_router/seal_rankings.py`、`score_evaluation.py` | 固定 30 条/复核 6 条实验实现，不能直接冒充通用算法 |
| 研究理想接触约束的可动范围 | `compiler.py`、`verified_interval.py` | 数学原型；真实视频的接触输入尚未可靠接通 |

详细输入、输出及限制见 [模块地图](docs/MODULES.md)。

## 五分钟运行一个可复用例子

在克隆后的 `quiethand-engineering/` 根目录，使用 Python 3.10 或更新版本：

```bash
python3 examples/reuse_demo.py
python3 -m unittest discover -s tests -p 'test_reuse_demo.py' -v
```

无需 GPU、网络、权重、数据集或第三方 Python 依赖。示例展示：

1. 把“正在操作的区域”明确绑定到一个物体 ID，再选对应 CAD，而不是按名字猜。
2. 无法确定的区域保留为空，不假装已经识别。
3. 将有来源、单位和坐标系的证据序列化。
4. 给一个人工构造的接触点计算约束区间。

这些输入明确标为 **synthetic 接口示例**，不是新跑出来的视频结果。想看真正的阶段结果，阅读 [结果与经验](docs/RESULTS_AND_LESSONS.md) 和 `records/evaluation_result.json`。

想在另一个项目中调用核心模块，可使用：

```bash
python3 -m pip install -e .
```

安装依赖、CPU 测试、历史 GPU 流水线的实际接入条件见 [运行与复用说明](docs/REUSE.md)。本次打包验证见 [验证记录](docs/VALIDATION.md)。

## 最值得带走的几条经验

- 先保证视频片段包含完整动作；原来的短窗曾把动作头尾误当成“没有动作”。
- 先对齐时间、米制尺度和坐标系，再算距离或判断轨迹漂不漂。
- 物体身份、手的左右、主动/支撑角色是三个不同问题，不能互相替代。
- 整段相对运动大，可能是真实动作，不一定是抖动；要分段看证据。
- “所有字段都有值”不是“所有内容都正确”；失败和弃权也是应该展示的输出。

## 跟别人有什么关系

NVIDIA Video to Data 和 Open-AoE 已覆盖很多基础流水线功能；DeMiAn、EgoLoc、AGILE 分别覆盖细粒度语言监督、交互时刻定位、几何约束优化。记录见 [相近工作对照](docs/RELATED_WORK.md)，核对日期为 2026-09-17，未进行同数据性能复现。

以后复用时，先按任务选择已有模块，或采用更完整的上游实现，不必为了保留本项目而继续维护所有旧组件。

## 包内与包外

包内是核心源码、历史 QuietHand 运行配方、选定测试、CPU 示例、少量既有文字/数值结果和说明。`CONTENTS.json` 列出准确来源路径；未修改历史科学结果。

原视频、深度、CAD、MANO/模型权重、第三方仓库、用户原始评语、旧网页媒体、服务器环境和其他课题不打包。它们仍保留在原工作区。数据与模型来源见 [第三方资产说明](THIRD_PARTY.md)。

代码的对外许可证尚未由所有者选定，建议先存入私有仓库。本文件夹是从原工作区分离的工程资产副本，只需上传这个文件夹内的内容，不要上传整个 autoresearch 工作区。

## GitHub 留档

个人私有仓库：[jojo8888883/quiethand-engineering](https://github.com/jojo8888883/quiethand-engineering)。脚本、文档、测试和记录按完整源码目录留档，可直接在线浏览或克隆使用。本机独立 Git 副本用于版本记录，不需要把个人 GitHub 授权给共享 Codex 账号。
