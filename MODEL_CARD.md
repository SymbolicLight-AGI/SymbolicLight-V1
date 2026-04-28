# SymbolicLight V1 Model Card

## Model Summary

SymbolicLight V1 is a spike-gated dual-path language model released as part of the SymbolicLight V1 open package.
This package contains a cleaned weights-only 0.8B checkpoint, tokenizer assets, model code, inference code, training scripts, and artifact-based verification materials.

The released checkpoint is intended as a pre-training and scale-up artifact.
It is not instruction-tuned, not RLHF/RLAIF-aligned, and should not be evaluated as a polished assistant model.

## Released Assets

- Model weights: `weights/pytorch/latest.pt`
- Tokenizer model: `tokenizer/sl_tokenizer.model`
- Tokenizer configuration: `tokenizer/tokenizer_config.json`
- Model implementation: `src/model.py`
- Inference and evaluation script: `src/eval_08.py`
- Training script: `src/train_base.py`
- Data pipeline code: `src/data_pipeline.py`

## License

The model weights, tokenizer assets, source code, training scripts, inference scripts, and public documentation are released under the Apache License, Version 2.0.
See `LICENSE`, `WEIGHTS_LICENSE.md`, and `NOTICE`.

Training and validation data are not released and are not licensed through this repository.

## Training Data Disclosure Boundary

The public package describes only aggregate data categories and mixture proportions.
It does not disclose raw training text, raw validation text, source-level dataset names, source identifiers, download URLs, or source-level manifests.

Users should prepare their own legally available corpora if they want to run the training pipeline.
The aggregate recipe in `src/train_base.py` is a domain-level template rather than a redistribution of the original corpus.

## Intended Use

- Research on sparse and spike-gated language model architectures
- Checkpoint loading and inference verification
- Reproducibility inspection of the released artifact boundary
- Smoke-test training with public or user-provided data
- Follow-up fine-tuning or evaluation under the user's own data and safety controls

## Out-of-Scope Use

- Treating the checkpoint as an instruction-following assistant
- Using the checkpoint for high-stakes decision-making without separate validation
- Assuming that public assets reconstruct the private pre-training corpus
- Redistributing data that is not included in this repository

## Known Limitations

- The checkpoint is pre-trained only and has no post-training alignment.
- The original raw corpus and source-level manifest are not public.
- The public package does not reproduce every paper table end to end.
- Factual generation can be unstable, especially for knowledge-intensive prompts.
- Safety behavior has not been tuned to modern assistant-model standards.

## Verification

The package includes `CHECKSUMS_SHA256.json` for file-level verification.
The recommended smoke-test commands are documented in `REPRODUCIBILITY.md`.
