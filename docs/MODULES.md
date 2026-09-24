# 模块地图

以下路径均相对仓库根目录。可移植部分在 `quiethand/`，历史 runner 在 `scripts/quiethand/`；后者保留调用顺序和实验配方，不构成统一 SDK 或可直接运行的 GPU 产品线。

## 可移植核心 API 与 CPU 检查

| 路径 / 接口 | 用途与边界 |
| --- | --- |
| [`quiethand/models.py`](../quiethand/models.py) | `Provenance`、`EvidenceRecord`、`EvidenceEvent`：保存值、状态、来源、单位、坐标系与时间；观测、推断、人工和派生证据分开。 |
| [`quiethand/schema/evidence_event.schema.json`](../quiethand/schema/evidence_event.schema.json) | 早期 `EvidenceEvent` 的字段规范，不代表所有历史 M3 JSON。 |
| [`quiethand/object_identity.py`](../quiethand/object_identity.py) | `parse_visual_binding`、`resolve_role_entity`：将区域 JSON 绑定到实体和正确 CAD。`tool`/`target` 是语义角色，不是 CAD 身份；未知为 `None`。当前契约要求两个区域对应不同实体。 |
| [`quiethand/m3_adapter_contract.py`](../quiethand/m3_adapter_contract.py)、[`quiethand/m3_v1_2_contract.py`](../quiethand/m3_v1_2_contract.py) | 检查分阶段输入输出、观察/弃权状态与完整动作粒度；换数据不能伪造历史来源。 |
| [`quiethand/geometry.py`](../quiethand/geometry.py)、[`quiethand/compiler.py`](../quiethand/compiler.py)、[`quiethand/verified_interval.py`](../quiethand/verified_interval.py) | 接触位置、法向和模式的数学约束与可动性区间；不能据此声称已从视频测得接触力。 |
| [`tests/`](../tests/) 与 [`examples/reuse_demo.py`](../examples/reuse_demo.py) | CPU 测试和已知 synthetic 示例；运行方式见[复用说明](REUSE.md)。 |

## 历史感知与实验配方

| 阶段 | 保留入口 | 所需资源 / 输出 |
| --- | --- | --- |
| 动作区间与语义 | [`scripts/quiethand/m3_v1_2/run_temporal_semantic.py`](../scripts/quiethand/m3_v1_2/run_temporal_semantic.py) | 时序 RGB 与 Qwen 资源，输出动作区间、左右手和物体语义。 |
| 抽帧与材料准备 | [`scripts/quiethand/m3_v1_2/materialize_temporal.py`](../scripts/quiethand/m3_v1_2/materialize_temporal.py)、[`scripts/quiethand/m3_v1_2/materialize_geometry.py`](../scripts/quiethand/m3_v1_2/materialize_geometry.py) | 区间、RGB-D、相机 `K`，输出显式源帧索引与模型输入。 |
| 手部恢复与米制对齐 | [`scripts/quiethand/m3_v1_2/run_raw_hand.py`](../scripts/quiethand/m3_v1_2/run_raw_hand.py)、[`scripts/quiethand/m3_5_fusion/run_raw_hand_metric.py`](../scripts/quiethand/m3_5_fusion/run_raw_hand_metric.py) | RGB、HaWoR/MANO、传感器内参及冻结输入，输出手网格与轨迹。 |
| 分割与物体位姿 | [`scripts/quiethand/m3_v1_2/run_perception.py`](../scripts/quiethand/m3_v1_2/run_perception.py)、[`scripts/quiethand/m3_jobs/runtime_common.py`](../scripts/quiethand/m3_jobs/runtime_common.py) | RGB、mask、传感器深度、`K`、CAD 与模型资源，输出 mask 和 `T_camera_object`。 |
| 实体绑定与融合 | [`scripts/quiethand/m3_5_fusion/repair_object_identity.py`](../scripts/quiethand/m3_5_fusion/repair_object_identity.py)、[`scripts/quiethand/m3_eval_router/build_evaluation_fusion.py`](../scripts/quiethand/m3_eval_router/build_evaluation_fusion.py) | 显式实体绑定、语义、手/物体结果，输出融合记录。 |

上述脚本需要原始输入清单、模型资源、数据访问和适配后的运行目录。仓库不含模型权重、数据、RGB-D、CAD 或完整实验资源；`prepare_*job*.py` 的参数只记录历史运行方式，不能直接当作部署命令。冻结版本可在 [`records/checkpoint_plan.json`](../records/checkpoint_plan.json) 查阅。

## 几何诊断与查看

- [`scripts/quiethand/m3_5_fusion/build_fusion_preview.py`](../scripts/quiethand/m3_5_fusion/build_fusion_preview.py)：深度反投影、相对运动与靠近证据；需要对齐帧、米制深度和 `K`。
- [`scripts/quiethand/m3_5_fusion/diagnose_trajectories.py`](../scripts/quiethand/m3_5_fusion/diagnose_trajectories.py)：坐标变换和参考轨迹误差；参考轨迹只可事后诊断，不能混入待评估预测。
- [`scripts/quiethand/m3_5_fusion/mask_pose_consistency.py`](../scripts/quiethand/m3_5_fusion/mask_pose_consistency.py)：投影 mesh 与 mask 的一致性检查；凸包近似不等于完整遮挡渲染或 3D 姿态真值。
- [`scripts/quiethand/serve_review.py`](../scripts/quiethand/serve_review.py)：带参数的本地预览服务；使用方式见[复用说明](REUSE.md)。

[`scripts/quiethand/m3_5_fusion/repair_hand_rgbd.py`](../scripts/quiethand/m3_5_fusion/repair_hand_rgbd.py)、`refine_spoon_*` 与根层 `run_m2_*`/`run_m3_*` 是特定案例、历史校验或早期流程，不应作为默认通用方法。它们保留用于理解历史边界，不改变已有科学结果。
