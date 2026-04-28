# SymbolicLight V1 Code Package

This repository is the public code package for **SymbolicLight V1**, a spike-gated dual-path language-model architecture.
It contains the Python implementation, tokenizer assets, training and inference entry points, documentation, and release metadata.

The 0.8B model weights and manuscript files are **not stored in this GitHub repository**.
They are intended to be distributed separately through an external model/artifact host.

## Package Layout

- `LICENSE`: Apache License, Version 2.0, covering the released code and public assets unless otherwise stated
- `MODEL_CARD.md`: model card and release-scope notes for SymbolicLight V1
- `WEIGHTS_LICENSE.md`: license scope for separately distributed cleaned weights
- `NOTICE`: copyright and release-boundary notice
- `THIRD_PARTY_NOTICES.md`: third-party dependency notice
- `src/`: public Python implementation, training loop, tokenizer tooling, and inference scripts
- `tokenizer/`: released tokenizer model, vocabulary, and configuration
- `artifacts/`: lightweight public smoke-test notes for the code package
- `docs/`: project lineage and 194M training narrative
- `REPRODUCIBILITY.md`: artifact-based reproducibility scope and verified commands
- `paper_results_manifest.json`: machine-readable summary of public and non-public result claims
- `train_runs_194m.json`: registry for the four main 194M runs and selected historical comparison checkpoints

## What This Repository Provides

- **Code path:** model definition, pre-training loop, tokenizer tooling, data pipeline, and inference helpers.
- **Tokenizer assets:** the public SL tokenizer files used by the released model family.
- **Training narrative:** documentation for the 194M controlled study and the 0.8B scale-up direction.
- **Reproducibility boundary:** smoke-test commands and manifests for what can and cannot be reproduced from this repository alone.

This repository should not be read as a complete model-weight release.
The 0.8B checkpoint is referenced by the documentation, but the binary weight file is intentionally distributed separately.

## Recommended Reading Order

1. [docs/model_lineage.md](docs/model_lineage.md)
2. [docs/194m_training_story.md](docs/194m_training_story.md)
3. [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
4. [MODEL_CARD.md](MODEL_CARD.md)

## Public Boundary

The GitHub repository includes code, tokenizer assets, and artifact-level documentation.
Training and validation corpora are not distributed with this repository; only aggregate data categories, mixture proportions, and preprocessing rules are documented.
Model weights and manuscript files are also not stored in this repository.

## License

The Apache-2.0 license applies to the released code, tokenizer assets, and public documentation in this repository.
It also applies to separately distributed cleaned SymbolicLight V1 weights unless their hosting page states otherwise.
It does not apply to training or validation corpora, which are not distributed with this repository.
The public data disclosure is limited to aggregate domain categories, mixture proportions, and preprocessing rules.
