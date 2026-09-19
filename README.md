# QuietHand · 阶段性工程资产

把第一视角操作视频里的动作语义、手部运动和物体状态整理到一起，并提供检查、纠错与结果查看工具。

这是 2026-09-19 整理的工程原型留档，方便未来接入其他项目。当前采用完整源码 ZIP 留档，源代码和详细文档都在包内，不是仅有一份项目介绍。

## 下载与运行

[下载完整工程包（ZIP）](./quiethand-engineering.zip)

下载后解压，进入 `quiethand-engineering/`，使用 Python 3.10 或更新版本：

```bash
python3 examples/reuse_demo.py
python3 -m unittest discover -s tests -p 'test_reuse_demo.py' -v
```

这个人工构造的接口示例不需要 GPU、网络、权重或数据集，展示物体身份绑定、带来源和坐标系的证据记录，以及理想接触约束区间。它不是新跑出来的视频实验结果。

## 包含什么

源码包保留 138 个文件及目录结构：

- `quiethand/`：统一字段、证据记录、物体身份绑定、约束计算等核心模块。
- `scripts/quiethand/`：Qwen3-VL、HaWoR、SAM2、FoundationPose 的历史运行配方，以及对齐、几何检查和网页查看工具。
- `tests/`、`examples/`：选定测试和可直接运行的 CPU 示例。
- `records/`：少量既有文字和数值结果。
- `docs/`：模块地图、运行说明、结果与经验、相近工作对照、打包验证记录。
- `CONTENTS.json`：准确的来源文件清单。

包内 README 是完整导航。打包时已在原工作区之外运行示例与 56 个测试，全部通过；离线 wheel 构建、隔离导入及 schema 资源读取也通过。

## 已跑通的主线

RGB 视频由 Qwen3-VL 提供动作区间和手部语义；HaWoR 提供手部网格与轨迹；SAM2 提供物体区域；FoundationPose 在传感器深度、相机内参和物体 CAD 的支持下估计位姿。随后对齐时间与坐标系、绑定实体、检查冲突和缺项，并生成查看页面及质检排序。

## 当前边界

这是工程资产，不是已经证明标注降本或机器人训练收益的方法，也不是普通 RGB 视频的一键标注产品。几何没有自动改对所有语义错误，末端没有接机器人控制器。

原视频、深度、CAD、MANO/模型权重、第三方仓库、用户原始评语、服务器环境和其他课题均不包含。第三方资产许可见包内 `THIRD_PARTY.md`。项目对外许可证尚未选定，本仓库用于私有留档。

包内 `docs/RELATED_WORK.md` 记录与 NVIDIA Video to Data、Open-AoE、DeMiAn、EgoLoc、AGILE 的关系，核对日期为 2026-09-17，未进行同数据性能复现。
