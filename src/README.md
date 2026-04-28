# SymbolicLight V1 Open Code

This directory contains the public Python implementation for SymbolicLight V1, including tooling for the released 0.8B checkpoint.

## Contents

- `model.py`: SymbolicLight model definition.
- `train_base.py`: distributed pre-training entry point with an aggregate domain-level data recipe.
- `pretokenize.py`: tokenization helper for local corpora.
- `eval_08.py`: PyTorch checkpoint evaluation and generation helper.
- `chat.py`: interactive single-process chat loop for local checkpoint testing.
- `data_pipeline.py`: aggregate-domain parquet and memmap data pipeline.
- `train_tokenizer.py`: tokenizer wrapper.

## Data Policy

This public package does not include raw training text, raw validation text, source-level dataset names, source identifiers, download URLs, or source-level manifests.
To train from scratch, prepare your own legally available corpus under the aggregate domain directories expected by `train_base.py`.

For a package-level reproducibility summary, see `../REPRODUCIBILITY.md` and `../paper_results_manifest.json`.

## License

The source code, training scripts, inference scripts, tokenizer assets, cleaned weights-only checkpoint, and public documentation are released under Apache License, Version 2.0, unless a file states otherwise.
Training and validation data are not included and are not licensed through this repository.

## Example

```bash
pip install -r requirements.txt

python eval_08.py \
  --checkpoint_path ../weights/pytorch/latest.pt \
  --generate_only \
  --prompts "SymbolicLight is" \
  --max_new_tokens 1 \
  --device cpu

python chat.py \
  --checkpoint_path ../weights/pytorch/latest.pt \
  --tokenizer_path ../tokenizer/sl_tokenizer.model \
  --device cuda \
  --allow_windows_cuda
```

For this pre-training checkpoint, `chat.py` uses a constrained decoding path and defaults to `--prompt_format answer`.
This reduces repetition and keeps replies closer to short direct answers, but it does not turn the checkpoint into a polished assistant model.

The checkpoint in `../weights/pytorch/latest.pt` is a pre-training checkpoint.
It is not instruction-aligned and should be evaluated as a scale-up/pre-training artifact rather than as a polished assistant model.

On Windows, the script uses a CPU-safe checkpoint loader by default. CUDA transfer for this large checkpoint is treated as experimental and requires `--allow_windows_cuda`.
