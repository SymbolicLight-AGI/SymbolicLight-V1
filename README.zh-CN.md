# SymbolicLight V1

这是 SymbolicLight V1 语言模型的**仅推理开源版本**。

- 论文：[arXiv:2605.21333](https://arxiv.org/abs/2605.21333)
- 权重：[SymbolicLight-AGI/SymbolicLight-V1](https://huggingface.co/SymbolicLight-AGI/SymbolicLight-V1)
- 英文说明：[README.md](README.md)

## 公开范围

本仓库包含：

- 加载公开权重所需的 0.8B 和 194M 模型定义；
- 最小 checkpoint 加载与文本生成示例；
- 对应的 tokenizer 运行时资产；
- [推理兼容性验证报告](INFERENCE_VERIFICATION.md)。

本仓库明确不包含：

- 预训练或微调入口；
- 数据管线、语料配方、source manifest 或预处理代码；
- tokenizer 训练代码；
- 训练、消融或实验启动脚本；
- 原始训练命令、优化器状态、训练数据或训练结果元数据。

公开的 `model.py` 用于说明模型结构并支持推理时前向计算，不提供私有训练管线。

## 目录

| 目录 | 用途 |
| --- | --- |
| [`model-08b/`](model-08b/README.zh-CN.md) | SymbolicLight V1 0.8B 模型和最小推理代码 |
| [`model-194m/`](model-194m/README.zh-CN.md) | SymbolicLight V1 194M 模型和最小推理代码 |
GitHub 仓库不存放权重。请从上述 Hugging Face 仓库下载权重，并将本地
checkpoint 路径传给推理脚本。

本版本支持 checkpoint 加载和推理复现，不承诺端到端复现训练过程或论文中的
训练结果。

## 许可证

除非文件另有说明，本仓库中的源码、tokenizer 资产、文档和发布元数据
采用 [Apache License 2.0](LICENSE)。训练及验证语料不包含在本仓库内，也不
通过本仓库授权。
