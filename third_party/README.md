# 第三方来源

本项目沿用 TurboVLA 的 Apache-2.0 代码和既有第三方声明。根目录 LICENSE 不会自动覆盖所有外部组件。

| 组件 | 来源 | 本仓库处理 |
| --- | --- | --- |
| TurboVLA | https://github.com/H-EmbodVis/TurboVLA | 模型、训练、数据、评估接口，保留原许可 |
| GroundingDINO | https://github.com/IDEA-Research/GroundingDINO | 融合/Transformer派生代码，保留文件头 |
| VLA-Adapter | https://github.com/OpenHelix-Team/VLA-Adapter | rollout派生代码，见licenses/VLA-Adapter.txt |
| StarVLA、OpenVLA、Diffusion Policy等 | 各子目录说明 | 保留原有licenses目录与文件头 |
| SDT-V3 | https://github.com/BICLab/Spike-Driven-Transformer-V3 | 下载上游模型文件，应用集成补丁 |
| SmoothSpike | https://github.com/CayleyZ/SmoothSpike | 下载上游运行代码，应用集成补丁 |

2026-09-07核对的 SmoothSpike 上游没有仓库级LICENSE；部分文件另带Apache或其他来源的版权声明。SDT-V3选用的模型文件也没有独立许可证头。本仓库因此只提交这两个模型的来源说明和集成补丁，通过下载脚本获取上游文件，不为它们补写或假定统一许可。使用、再分发前请核对上游条款。

`scripts/setup_third_party.py` 从上游main分支获取5个文件，并应用 `patches/integration.patch`。补丁保留实际实验中的改动：

- SmoothSpike包内相对导入；
- DDP设备放置前在CPU构造旋转矩阵；
- 支持2D、3D、4D文本mask；
- 冻结H1时缓存矩阵符号变换；
- 将冻结推理未使用的fast Hadamard扩展改为可选导入；真正调用该算子仍明确报缺失依赖；
- 移除SDT-V3中未使用的torchinfo导入。

这些是集成修改，没有替换或简化模型主体。模型权重、tokenizer和数据均从相应上游另行获取。
