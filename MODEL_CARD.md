# SymbolicLight V1 Model Card

## Model Summary

SymbolicLight V1 is a spike-gated dual-path language model.
This GitHub repository contains the public code package: model implementation, tokenizer assets, training scripts, inference scripts, and reproducibility notes.

The cleaned 0.8B weights are not stored in this GitHub repository.
They are intended to be distributed separately through an external model/artifact host.
When used, the released checkpoint should be treated as a pre-training and scale-up artifact rather than as an instruction-following assistant model.

## Released Assets In This Repository

- Tokenizer model: `tokenizer/sl_tokenizer.model`
- Tokenizer configuration: `tokenizer/tokenizer_config.json`
- Model implementation: `src/model.py`
- Inference and evaluation script: `src/eval_08.py`
- Training script: `src/train_base.py`
- Data pipeline code: `src/data_pipeline.py`
- Reproducibility notes: `REPRODUCIBILITY.md`

## Separately Distributed Assets

- Cleaned weights-only 0.8B checkpoint
- Manuscript PDF and submission files
- External archive metadata, if published

## License

The tokenizer assets, source code, training scripts, inference scripts, and public documentation in this repository are released under the Apache License, Version 2.0.
Separately distributed cleaned SymbolicLight V1 weights are also intended to use Apache-2.0 unless their hosting page states otherwise.
See `LICENSE`, `WEIGHTS_LICENSE.md`, and `NOTICE`.

Training and validation data are not released and are not licensed through this repository.

## Training Data Disclosure Boundary

The public package describes only aggregate data categories and mixture proportions.
It does not disclose raw training text, raw validation text, source-level dataset names, source identifiers, download URLs, or source-level manifests.

Users should prepare their own legally available corpora if they want to run the training pipeline.
The aggregate recipe in `src/train_base.py` is a domain-level template rather than a redistribution of the original corpus.

## Intended Use

- Research on sparse and spike-gated language model architectures
- Inspection of SymbolicLight V1 model code and training code
- Smoke-test training with public or user-provided data
- Loading separately distributed SymbolicLight V1 checkpoints
- Follow-up fine-tuning or evaluation under the user's own data and safety controls

## Out-of-Scope Use

- Treating the pre-training checkpoint as an instruction-following assistant
- Using the checkpoint for high-stakes decision-making without separate validation
- Assuming that public assets reconstruct the private pre-training corpus
- Redistributing data that is not included in this repository

## Known Limitations

- The 0.8B checkpoint is pre-trained only and has no post-training alignment.
- The original raw corpus and source-level manifest are not public.
- This GitHub repository does not include the model weight binary.
- The public package does not reproduce every paper table end to end.
- Factual generation can be unstable, especially for knowledge-intensive prompts.
- Safety behavior has not been tuned to modern assistant-model standards.
