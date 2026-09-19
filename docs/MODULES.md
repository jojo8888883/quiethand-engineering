# 模块地图：以后需要什么，就找哪一块

所有路径相对解压后的包根目录。`quiethand/` 是可导入的 Python 包；`scripts/quiethand/` 保留原配方和实验脚本，不是统一 SDK。

## 可直接调用的核心

| 文件 / 接口 | 输入 → 输出 | 必须保留的含义 |
| --- | --- | --- |
| `quiethand/models.py`：`Provenance`、`EvidenceRecord`、`EvidenceEvent` | 值、状态、来源、单位、坐标系、时间 → 可校验/序列化记录 | 缺失原因显式保存；观测、推断、人工、派生证据分开 |
| `quiethand/schema/evidence_event.schema.json` | 记录字段规范 | 是早期 EvidenceEvent 格式，不代表所有 M3 JSON 都使用同一 schema |
| `quiethand/object_identity.py`：`parse_visual_binding`、`resolve_role_entity` | 两个视觉区域的 JSON 绑定 + 物体目录 → `role_to_entity` + 正确 mesh 条目 | `tool`/`target` 是语义区域，不是 CAD 身份；未知返回 `None` |
| `quiethand/m3_adapter_contract.py` | 分阶段输入、结果文件及观察/弃权状态 → 输入输出契约检查 | 是冻结实验的严格接口，换数据不应伪造旧实验来源 |
| `quiethand/m3_v1_2_contract.py` | 完整动作区间与结果 → v1.2 格式检查 | 完整动作粒度；不是固定两秒窗 |
| `quiethand/geometry.py`、`compiler.py`、`verified_interval.py` | 接触位置、法向、模式/区间 → 数学约束与方向可动性区间 | 单位和坐标约定明确；不能据此宣称已从视频测出接触力 |

`object_identity` 的现有约束是两个不同区域对应两个不同实体，**不支持两只手共同抓同一物体的通用场景**。以后该场景需要新的真实接口设计，不能直接套用旧约束。

## 感知与对齐：原流水线入口

| 阶段 | 主入口 | 依赖与输出 |
| --- | --- | --- |
| 完整动作定位/语义 | `scripts/quiethand/m3_v1_2/run_temporal_semantic.py` | 时序 RGB、Qwen 资源 → 动作区间、左右手和物体语义 |
| 抽帧与材料准备 | `m3_v1_2/materialize_temporal.py`、`materialize_geometry.py` | 区间、原 RGB-D 与相机参数 → 显式源帧索引和模型输入 |
| 原始手 | `m3_v1_2/run_raw_hand.py` | RGB、HaWoR/MANO → 手网格与轨迹 |
| 米制手对齐 | `m3_5_fusion/prepare_metric_hand_input.py`、`run_raw_hand_metric.py`、`audit_metric_alignment.py` | 传感器内参与冻结输入 → 同相机模型下的手结果；并非任意轨迹都准确 |
| 物体分割/位姿 | `m3_v1_2/run_perception.py`，公共运行工具为 `m3_jobs/runtime_common.py` | RGB、mask、sensor depth、K、CAD → mask 与 `T_camera_object` |
| 物体身份修正 | `m3_5_fusion/repair_object_identity.py` | 显式实体绑定 → 选择对应 mesh 重估或复用姿态 |
| evaluation 记录构建 | `m3_eval_router/build_evaluation_fusion.py` | 语义、身份、手、物体结果 → 30 条融合记录 |

这些脚本依赖真实资源和原运行目录。`prepare_*job*.py` 等任务生成器含原服务器绝对路径；列为**需要接入的运行配方**，不是直接执行按钮。包内保留源码便于查调用顺序，没有执行、替换或上传任何模型。

## 几何检查与诊断

| 接口 | 用途 | 可用范围 |
| --- | --- | --- |
| `m3_5_fusion/build_fusion_preview.py`：`visible_surface_points`、`relative_envelope`、`pair_evidence` | 深度反投影、手物相对运动、靠近证据 | 需正确帧对齐、米制深度、相机参数；其中批处理 `main` 是旧数据配方 |
| `m3_5_fusion/diagnose_trajectories.py`：`transform_points`、`trajectory_errors`、`relative_positions` | 坐标变换与参考轨迹误差 | 参考轨迹只能用于事后检查，不能混进待评估预测 |
| `m3_5_fusion/diagnose_hand_object_stability.py` | 预测手/参考手 × 预测物体/参考物体的四种组合 | 定位是哪条支路造成异常；批处理与案例绑定 |
| `m3_5_fusion/mask_pose_consistency.py`：`projected_mask_iou` | 将 mesh 投影与 mask 比较 | 使用投影凸包近似，不是完整遮挡感知渲染，不等于 3D 姿态真值 |
| `mask_depth_translation.py`、`mask_depth_se3_icp.py` | 深度约束的平移或 SE(3) 修正 | 保留方法与测试；不作为默认可靠修复器，见失败记录 |

## 展示、人工反馈与质检

- `m3_v1_2/build_result_preview.py`、`review.js`：中文语义/视频查看与意见导出。
- `m3_v1_2/build_correction_review.py`：整理拒绝记录与自然语言修改意见。
- `m3_5_fusion/build_identity_batch_preview.py`、`identity_batch_preview.html`：几何前后对照。
- `serve_review.py`：本地 HTTP byte-range，解决视频拖动/播放所需的请求。
- `m3_eval_router/seal_rankings.py`：按原冻结规则排序。
- `m3_eval_router/score_evaluation.py`：在相同查看预算下计分。

## 不要当成默认通用方法

- `entity_semantic_binding.py`：固定单个事件/两个 CAD，终态是角色反转失败。
- `repair_hand_rgbd.py` 和 `refine_spoon_*`：特定案例诊断/修复探索，有失败或局限；不是全数据自动修复。
- `build_role_swap_symmetry.py`：交换的是结构化字段，未新增真实左手主导视频或独立标签。
- 根层 `run_m2_*`、`run_m3_*`：早期数据落地、校验或短窗历史流程。留档可查，但重新接视频优先看完整动作的 v1.2 路线。

这次没有统一改写历史 API，以免破坏已保存结果。新的项目应从所需核心函数接入自己的输入，而不是先把所有历史 gate 重走一遍。
