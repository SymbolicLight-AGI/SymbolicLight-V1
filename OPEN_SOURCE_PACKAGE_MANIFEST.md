# SymbolicLight V1 Open Package

Prepared for release review on 2026-04-26.

## Included

- `LICENSE`: Apache License, Version 2.0.
- `WEIGHTS_LICENSE.md`: release scope for model weights and tokenizer assets.
- `MODEL_CARD.md`: model card for the released SymbolicLight V1 checkpoint.
- `NOTICE`: release boundary and copyright notice.
- `THIRD_PARTY_NOTICES.md`: third-party dependency notice.
- `README.md`: public repository root and unified release narrative.
- `src/`: Python model, pre-training loop, data pipeline, tokenizer wrapper, and generation utility.
- `tokenizer/`: tokenizer model, vocabulary, and tokenizer configuration.
- `weights/pytorch/latest.pt`: current 0.8B PyTorch checkpoint.
- `paper/`: unified preprint source, compiled PDF, and bibliography output.
- `docs/`: historical lineage and 194M training-story documentation.
- `REPRODUCIBILITY.md`: public artifact-based reproducibility guide.
- `paper_results_manifest.json`: machine-readable summary of what is and is not publicly reproducible.
- `train_runs_194m.json`: registry for the four main 194M runs and related historical comparison checkpoints.
- `artifacts/`: lightweight smoke-test logs and checkpoint metadata summaries.

## Excluded

- Raw training text and raw validation text.
- Source-level dataset names, source identifiers, download URLs, and source-level manifests.
- Local training-data directories such as `data_bin`, `sft_data`, `.hf_cache`, and private parquet corpora.
- Third-party baseline model weights and local test-weight directories.
- C++ acceleration engine source, C++ export scripts, and C++ INT8 inference weights.
- Historical SFT checkpoints, ablation checkpoints, temporary outputs, cache files, and build artifacts.
- Earlier paper drafts that expose concrete source-level data names.
- Legacy DualPath-style public skeletons whose directory layout no longer matches the current manuscript.

## Public Data Policy

The package exposes model code, training scripts, inference scripts, tokenizer files, weights, and a historical run registry.
It intentionally does not expose raw data or source-level manifests.
Users should train or audit the pipeline using their own legally available corpora organized under the aggregate domain recipe in `src/train_base.py`.

## License Policy

The code, training scripts, inference scripts, tokenizer assets, cleaned weights-only checkpoint, documentation, and public manifests are released under Apache License, Version 2.0, unless a file states otherwise.
The training and validation data are excluded from this license because they are not distributed in the package.
The public data disclosure is limited to aggregate domain types, mixture proportions, preprocessing rules, and artifact-level verification metadata.

## Review Before Upload

- Confirm that the Apache-2.0 release scope remains correct before publishing.
- Confirm company ownership and authority to release the checkpoint.
- Re-run the sensitive-name scan before uploading to GitHub, HuggingFace, Zenodo, or another public host.
- Confirm that the published checkpoint remains the cleaned weights-only checkpoint and does not include optimizer state or data-loader state.
