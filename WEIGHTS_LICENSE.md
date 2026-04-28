# SymbolicLight V1 Weights License

The released SymbolicLight V1 code package in this repository is licensed under the Apache License, Version 2.0, unless a file states otherwise.

This applies to:

- Source code under `src/`
- Training scripts and inference scripts
- Tokenizer assets under `tokenizer/`
- Public documentation and verification manifests in this repository

The cleaned SymbolicLight V1 weights are not stored in this GitHub repository.
When those weights are distributed separately by the project, they are intended to be released under Apache License, Version 2.0, unless the external hosting page states otherwise.

## Data Exclusion

The license does not grant rights to any training or validation data used to train the model.
Raw training text, raw validation text, source-level dataset names, source identifiers, download URLs, and source-level manifests are not included in this repository and are not licensed for redistribution.

The public data description is limited to aggregate domain categories, mixture proportions, preprocessing rules, and artifact-level verification metadata.
Users who train or fine-tune the model are responsible for using corpora that they have the right to process and redistribute.

## Checkpoint Scope

The separately distributed checkpoint should be a cleaned weights-only release.
It should not include optimizer state, mixed-precision scaler state, data-loader state, local filesystem paths, or source-level data manifests.

## Citation

If you use this model or code in research, please cite the accompanying SymbolicLight V1 paper and repository when citation metadata is available.
