# SymbolicLight V1 194M

本包包含公开的 194M 模型定义、tokenizer 运行时和最小文本生成入口。
checkpoint 单独托管在
[Hugging Face](https://huggingface.co/SymbolicLight-AGI/SymbolicLight-V1)。

## 推理

```bash
python -m pip install -r requirements.txt
python code/inference.py --checkpoint /path/to/symboliclight-v1-194m-inference.pt --prompt "Once upon a time"
```

194M checkpoint 使用独立的 48K tokenizer，请勿与 0.8B tokenizer 混用。

## 发布边界

这里只公开模型构造和推理工具，不包含训练入口、数据管线、预处理、语料配方、
tokenizer 训练、评测启动器、消融脚本或原始训练命令。
