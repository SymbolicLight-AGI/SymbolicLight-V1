# SymbolicLight V1 0.8B

This package contains the public 0.8B model definition, tokenizer runtime, and
a minimal text-generation entry point. The checkpoint is hosted separately on
[Hugging Face](https://huggingface.co/SymbolicLight-AGI/SymbolicLight-V1).

## Inference

Install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate text with a downloaded checkpoint:

```bash
python src/inference.py --checkpoint /path/to/latest-inference.pt --prompt "Once upon a time"
```

The public tokenizer has 55,296 active tokens inside the model's 57,344-entry
structural vocabulary. Do not use the 194M tokenizer with this checkpoint.

## Release boundary

Only model construction and inference utilities are public. Training entry
points, data pipelines, preprocessing, corpus recipes, tokenizer training,
experiment launchers, and original training commands are not included.
