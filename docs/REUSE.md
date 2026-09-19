# 运行与复用说明

## 1. 最短路径：先跑纯 CPU 接口示例

解压 `quiethand-engineering.zip`，终端进入解压后的 `quiethand-engineering/`，执行：

```bash
python3 examples/reuse_demo.py
python3 -m unittest discover -s tests -p 'test_reuse_demo.py' -v
python3 -m unittest discover -s tests -p 'test_quiethand_m[01].py' -v
```

Python >= 3.10。以上不需要安装依赖，也不会调用模型或写入旧结果。示例的 JSON 打印到终端。

从其他项目 import 核心包：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

`pyproject.toml` 仅负责核心 Python 包。可选的 CPU 数值/图像模块和随包测试：

```bash
.venv/bin/python -m pip install -e '.[cpu]'
.venv/bin/python -m unittest discover -s tests -v
```

这不是 GPU 环境安装命令；没有把 CUDA、FoundationPose 和 HaWoR 强行塞进一个通用 requirements。

## 2. 用自己的物体身份输入

```python
from quiethand.object_identity import parse_visual_binding, resolve_role_entity

event = {
    'event_id': 'your-event',
    'object_state': {'entities': [
        {'entity_id': 'bowl', 'mesh_path': 'meshes/bowl.obj'},
        {'entity_id': 'plate', 'mesh_path': 'meshes/plate.obj'},
    ]},
}
raw = '''{
  "region_0": {"entity_id": "bowl", "reason": "区域内是深碗"},
  "region_1": {"entity_id": "plate", "reason": "区域内是浅盘"}
}'''
binding = parse_visual_binding(raw, event)
mesh_entry = resolve_role_entity(event, binding, 'tool')
print(mesh_entry)
```

这里的 JSON 是调用者提供的已获得回答；这段代码本身不识别图像、不生成 CAD，也不判断主动/支撑。两个区域当前要求指向不同实体，未知可填 `null`。

## 3. 接回实际感知流水线

先准备的输入：RGB、与 RGB 对应的传感器深度、内参、物体 CAD，以及每个事件的源帧索引。现有协议中几何常用完整动作里的 15 个采样帧；稀疏帧不能直接当高频控制轨迹。

顺序：

1. 用完整视频确定动作区间与语义。
2. 按同一组源帧索引准备 RGB-D 与相机参数。
3. 生成物体区域并显式绑定物体身份。
4. 跑手与物体位姿，先确认尺度/坐标一致，再计算相对关系。
5. 输出原始观察、冲突和缺项；用预览查看，不把推断伪装成真值。

实际入口见 `docs/MODULES.md`；历史模型版本见 `records/checkpoint_plan.json`。模型 runner 的参数由相邻 `prepare_*job*.py` 生成。运行时还需要原输入清单、绑定记录和资源目录，这些不是凭空用一个空 JSON 就能替代的。

原脚本里的服务器路径和实验 ID 是历史运行配方的一部分。迁入新项目时，为那个项目提供明确输入、模型目录和输出路径；不要照抄排卡命令或直接重放已完成实验。本包没有宣称 GPU 全链路可以离开原资源立即复现。

## 4. 看旧网页

包里有网页生成器，但没有旧视频和点云。要看原结果，应在原工作区或已有本地预览站点启动视频服务：

```bash
python3 scripts/quiethand/serve_review.py --bind 127.0.0.1 --port 8768 --directory .
```

选择保留媒体的目录作为 `--directory`，否则 HTML 能打开但视频仍会 404。不要把“网页存在”当成“媒体也在包里”。

## 5. 硬件与环境边界

- 本包接口例子：普通 CPU，macOS/Linux 可用，无 GPU。
- 随包几何模块与测试：CPU + NumPy/Pillow/SciPy/imageio-ffmpeg，已列在 `cpu` 可选依赖中；其他历史入口按实际需求配置。
- 历史模型全链路：分离的 Qwen、HaWoR、SAM2/FoundationPose 环境，原来在共享 A800 上执行；本轮未测最低显存，也未新建环境。
- 数据/模型需按各自官方来源获取。ARCTIC、MANO 等访问许可不能从本包继承。

## 6. 原工作区重新导出

```bash
python3 scripts/quiethand/export_engineering_bundle.py
```

输出 `deliverables/quiethand-engineering.zip`。导出器只读取指定源码、本文档目录和选定记录，不访问服务器、不调用模型，不复制原始数据或用户评语。`CONTENTS.json` 可用于查一个文件来自哪里。
