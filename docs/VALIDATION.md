# 验证记录

## 2026-09-24：公开仓库整理

本次整理了 README、模块导航、源码克隆与运行说明、脚本分类和包元数据；未修改模型实现、历史实验配方或已有数值结果。

| 检查 | 环境与结果 |
| --- | --- |
| `examples/reuse_demo.py` | Python 3.13.2；成功，输出保持 `synthetic_interface_only` 与 `model_called=false` |
| 全部 CPU 单元测试 | 已有数值依赖环境；56 / 56 通过，约 6 秒 |
| 核心 wheel | 离线构建成功，不下载依赖 |
| 独立安装 | 在新建的隔离 venv 中安装 wheel；从仓库外以 Python 隔离模式导入核心接口、读取并解析 JSON schema 成功 |
| 文档导航 | 仓库内 Markdown 相对链接目标存在 |

测试覆盖物体身份绑定、缺项与来源记录、约束数学、相对运动、投影/mask 一致性、合成几何修正和接口示例。已有测试中的一个正则字符串产生 Python SyntaxWarning，但不影响测试通过。

本轮验证的是核心包与 CPU 接口，不是从零重建全部数值依赖或 GPU 环境；没有重跑 Qwen、HaWoR、SAM2、FoundationPose，也没有新增效果、降本或机器人结果。历史脚本的原目录与资源依赖见[脚本索引](../scripts/quiethand/README.md)。

## 2026-09-19：初次工程整理

初次整理已在原工作区之外验证示例、56 项 CPU 测试、离线 wheel 构建与隔离安装，并检查所选文件不包含原始媒体、模型权重或用户原始评语。[CONTENTS.json](../CONTENTS.json) 保留该次整理的来源索引；当前版本的文件以 Git 跟踪内容为准。
