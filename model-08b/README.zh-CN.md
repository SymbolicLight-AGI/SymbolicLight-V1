# SymbolicLight V1 0.8B

本包包含公开的 0.8B 模型定义、tokenizer 运行时和最小文本生成入口。
checkpoint 单独托管在
[Hugging Face](https://huggingface.co/SymbolicLight-AGI/SymbolicLight-V1)。

## 推理

```bash
python -m pip install -r requirements.txt
python src/inference.py --checkpoint /path/to/latest-inference.pt --prompt "Once upon a time"
```

公开 tokenizer 在模型 57,344 个结构词表槽位中使用 55,296 个有效 token。
请勿将 194M tokenizer 与该 checkpoint 混用。

## 发布边界

这里只公开模型构造和推理工具，不包含训练入口、数据管线、预处理、语料配方、
tokenizer 训练、实验启动脚本或原始训练命令。
