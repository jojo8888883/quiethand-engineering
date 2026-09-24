# 运行与复用说明

QuietHand 的公开仓库是 [jojo8888883/quiethand-engineering](https://github.com/jojo8888883/quiethand-engineering)，项目与视频说明见 [项目页面](https://quiethand-brief-yuzhou.scut-lang.chatgpt.site/)。本仓库可直接复用的是核心 Python API、CPU 几何/契约检查和本地预览服务；历史模型 runner 与冻结实验配方仅供查阅，不是开箱即用的 GPU 全链路。

## 最短可运行路径

在任意新目录克隆仓库。需要 Python 3.10 或更高版本：

```bash
git clone https://github.com/jojo8888883/quiethand-engineering.git
cd quiethand-engineering
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[cpu]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python examples/reuse_demo.py
```

`examples/reuse_demo.py` 是已知的 synthetic 接口示例：它演示物体身份绑定、证据记录和接触约束计算，不读取视频、不下载资源，也不代表模型或实验结果。`cpu` 额外依赖只覆盖 NumPy、Pillow、SciPy 与 imageio-ffmpeg；这不是 CUDA 或模型环境安装命令。

可在其他项目中直接导入核心包：

```python
from quiethand.object_identity import parse_visual_binding, resolve_role_entity
from quiethand.models import EvidenceRecord
```

接口与文件用途见[模块地图](MODULES.md)。

## 接入真实感知输入时的边界

历史感知路线需要逐事件准备：时序 RGB、与 RGB 对齐的传感器深度、相机内参 `K`、物体 CAD、完整动作区间和源帧索引。完整动作的几何处理曾使用 15 个采样帧；稀疏帧不能直接当作高频控制轨迹。

典型顺序是：先由完整视频确定动作区间和语义；按同一源帧索引准备 RGB-D、`K` 与 CAD；再运行手和物体估计、确认尺度与坐标系一致，最后生成带缺项/冲突状态的结构化记录。核心绑定 API 只解析调用者已经获得的区域—实体 JSON；它不识别图像、不生成 CAD，也不推断主动或支撑角色。

[`scripts/quiethand/m3_v1_2/`](../scripts/quiethand/m3_v1_2/) 和 [`scripts/quiethand/m3_5_fusion/`](../scripts/quiethand/m3_5_fusion/) 中保留了 Qwen、HaWoR、SAM2、FoundationPose 等历史 runner、任务准备脚本与冻结实验配方。它们仍要求各自的模型、数据、运行环境与输入目录；这些资源未随仓库提供。本仓库没有提供通用 GPU 安装器，也未声明任意机器可复现完整 GPU 流水线。

模型、数据集和部分人体模型还受各自的获取条件约束，不能因克隆本仓库而获得。具体归属见[第三方资产说明](../THIRD_PARTY.md)。

## 本地查看已有预览

[`scripts/quiethand/serve_review.py`](../scripts/quiethand/serve_review.py) 是参数化的本地 HTTP 服务，支持视频 byte-range 请求。将 `--directory` 指向你已有的 HTML 与媒体目录：

```bash
.venv/bin/python scripts/quiethand/serve_review.py \
  --bind 127.0.0.1 --port 8768 --directory path/to/review-artifacts
```

它不会生成历史视频、点云或实验输出；目录中缺失的媒体仍会返回 404。
