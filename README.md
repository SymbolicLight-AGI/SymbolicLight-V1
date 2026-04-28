# SymbolicLight V1 Open Package

This repository is the public release root for **SymbolicLight V1**.
It combines two parts of the project narrative in one place:

1. The **194M Dual-Path pre-training story**, which provides the main controlled language-modeling evidence.
2. The **0.8B scale-up release**, which provides a public checkpoint, tokenizer, executable code paths, and artifact-based verification material.

The repository should therefore be read as a **unified public package**, not as two separate model lines.
The architecture name used throughout the current paper, code, and public materials is **SymbolicLight V1**.

## Package Layout

- `LICENSE`: Apache License, Version 2.0, covering the released code and public assets unless otherwise stated
- `WEIGHTS_LICENSE.md`: license scope for the cleaned model weights and tokenizer assets
- `MODEL_CARD.md`: model card for the released SymbolicLight V1 checkpoint
- `NOTICE`: copyright and release-boundary notice
- `THIRD_PARTY_NOTICES.md`: third-party dependency notice
- `src/`: public Python implementation, training loop, tokenizer tooling, and inference scripts
- `tokenizer/`: released tokenizer model, vocabulary, and configuration
- `weights/pytorch/latest.pt`: released weights-only 0.8B checkpoint
- `paper/`: unified English manuscript, Chinese companion manuscript, bibliography, and compiled PDFs
- `artifacts/`: public smoke-test logs and checkpoint metadata summaries
- `docs/`: project lineage, 194M training narrative, and release-facing documentation
- `REPRODUCIBILITY.md`: artifact-based reproducibility scope and verified commands
- `paper_results_manifest.json`: machine-readable summary of public and non-public result claims
- `train_runs_194m.json`: registry for the four main 194M runs and selected historical comparison checkpoints

## What This Package Tries To Tell

The intended public narrative is:

- **194M SymbolicLight V1** is the main controlled study.
  It establishes that a spike-gated dual-path language model can train stably at high activation sparsity and remain competitive with dense baselines.
- **0.8B SymbolicLight V1** is scale-up evidence.
  It shows that the same overall architectural direction can be extended to a larger native pre-training run.
- The public release supports **artifact inspection**, **checkpoint verification**, **inference verification**, and **smoke-test training**.
  It does **not** provide full public reconstruction of the original private pre-training corpus.

## Recommended Reading Order

1. [docs/model_lineage.md](docs/model_lineage.md)
2. [docs/194m_training_story.md](docs/194m_training_story.md)
3. [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
4. [paper/SymbolicLight_V1_unified_preprint.pdf](paper/SymbolicLight_V1_unified_preprint.pdf)

## Public Boundary

The release includes code, tokenizer assets, cleaned model weights, and artifact-level documentation.
Training and validation corpora are not distributed with this repository; only aggregate data categories, mixture proportions, and preprocessing rules are documented.

## License

The Apache-2.0 license applies to the released code, tokenizer assets, cleaned model weights, and public documentation.
It does not apply to training or validation corpora, which are not distributed with this repository.
The public data disclosure is limited to aggregate domain categories, mixture proportions, and preprocessing rules.
