# SymbolicLight V1

Inference-only open release for the SymbolicLight V1 language models.

- Paper: [arXiv:2605.21333](https://arxiv.org/abs/2605.21333)
- Weights: [SymbolicLight-AGI/SymbolicLight-V1](https://huggingface.co/SymbolicLight-AGI/SymbolicLight-V1)
- Chinese documentation: [README.zh-CN.md](README.zh-CN.md)

## Public scope

This repository contains:

- the 0.8B and 194M model definitions required to load the public weights;
- minimal checkpoint-loading and text-generation examples;
- the corresponding tokenizer runtime assets;
- an [inference compatibility report](INFERENCE_VERIFICATION.md).

This repository intentionally does **not** contain:

- pre-training or fine-tuning entry points;
- data pipelines, corpus recipes, source manifests, or preprocessing code;
- tokenizer-training code;
- training, ablation, or experiment-launch scripts;
- original training commands, optimizer state, training datasets, or training-result metadata.

The public `model.py` files describe the model architecture and inference-time
forward path. They do not provide the private training pipeline.

## Packages

| Package | Purpose |
| --- | --- |
| [`model-08b/`](model-08b/README.md) | SymbolicLight V1 0.8B model and minimal inference |
| [`model-194m/`](model-194m/README.md) | SymbolicLight V1 194M model and minimal inference |
Weights are intentionally excluded from GitHub. Download them from the linked
Hugging Face repository and supply their local paths to the inference scripts.

This release supports checkpoint loading and inference reproduction. It does
not claim end-to-end reproduction of training or the paper's training results.

## License

Unless a file states otherwise, source code, tokenizer assets, documentation,
and release metadata in this repository are licensed under the
[Apache License 2.0](LICENSE). Training and validation corpora are not included
and are not licensed by this repository.
