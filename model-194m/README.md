# SymbolicLight V1 194M

This package contains the public 194M model definition, tokenizer runtime, and
a minimal text-generation entry point.
Checkpoints are hosted separately on
[Hugging Face](https://huggingface.co/SymbolicLight-AGI/SymbolicLight-V1).

## Inference

```bash
python -m pip install -r requirements.txt
python code/inference.py --checkpoint /path/to/symboliclight-v1-194m-inference.pt --prompt "Once upon a time"
```

The 194M checkpoint uses its own 48K tokenizer. Do not use the 0.8B tokenizer
with these weights.

## Release boundary

Only model construction and inference utilities are public. Training entry
points, data pipelines, preprocessing, corpus recipes, tokenizer training,
evaluation launchers, ablation scripts, and original training commands are not
included.
