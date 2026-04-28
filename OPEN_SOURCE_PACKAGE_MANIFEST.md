# SymbolicLight V1 GitHub Code Package

Prepared for public GitHub release.

## Included

- `LICENSE`: Apache License, Version 2.0.
- `WEIGHTS_LICENSE.md`: release scope for separately distributed model weights and tokenizer assets.
- `MODEL_CARD.md`: model card and release-scope notes for SymbolicLight V1.
- `NOTICE`: release boundary and copyright notice.
- `THIRD_PARTY_NOTICES.md`: third-party dependency notice.
- `README.md`: public repository root and unified release narrative.
- `src/`: Python model, pre-training loop, data pipeline, tokenizer wrapper, and generation utility.
- `tokenizer/`: tokenizer model, vocabulary, and tokenizer configuration.
- `docs/`: historical lineage and 194M training-story documentation.
- `REPRODUCIBILITY.md`: public artifact-based reproducibility guide.
- `paper_results_manifest.json`: machine-readable summary of what is and is not publicly reproducible.
- `train_runs_194m.json`: registry for the four main 194M runs and related historical comparison checkpoints.
- `artifacts/`: lightweight smoke-test notes and release-facing artifact notes.

## Excluded From GitHub

- Cleaned 0.8B checkpoint binary.
- Manuscript source, bibliography, and PDF files.
- Raw training text and raw validation text.
- Source-level dataset names, source identifiers, download URLs, and source-level manifests.
- Local training-data directories such as `data_bin`, `sft_data`, `.hf_cache`, and private parquet corpora.
- Third-party baseline model weights and local test-weight directories.
- C++ acceleration engine source, C++ export scripts, and C++ INT8 inference weights.
- Historical SFT checkpoints, ablation checkpoints, temporary outputs, cache files, and build artifacts.
- Earlier paper drafts that expose concrete source-level data names.

## Public Data Policy

The GitHub package exposes model code, training scripts, inference scripts, tokenizer files, documentation, and a historical run registry.
It intentionally does not expose raw data or source-level manifests.
Users should train or audit the pipeline using their own legally available corpora organized under the aggregate domain recipe in `src/train_base.py`.

## License Policy

The code, training scripts, inference scripts, tokenizer assets, documentation, and public manifests in this repository are released under Apache License, Version 2.0, unless a file states otherwise.
Separately distributed cleaned SymbolicLight V1 weights are intended to use Apache-2.0 unless their hosting page states otherwise.
The training and validation data are excluded from this license because they are not distributed in the package.
The public data disclosure is limited to aggregate domain types, mixture proportions, preprocessing rules, and artifact-level verification metadata.
