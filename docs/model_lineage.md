# Model Lineage

This document explains how the current public package connects the earlier SymbolicLight releases into one coherent line.

## Unified Naming

The public-facing architecture name is **SymbolicLight V1**.

Earlier internal or preprint-stage labels such as "DualPath", "V2", "V3", or "V4" should not be read as separate production architectures in the current release package.
They mainly reflect earlier manuscript states, scaling stages, or historical packaging decisions.

## Lineage Summary

### Early prototype stage

- Sparse TCAM lookup
- LIF-based spike encoding
- surrogate-gradient training
- proof-of-concept language-modeling experiments at smaller scale

Representative reference:

- Liu, T. (2026). *SymbolicLight: A neuro-symbolic spiking architecture for language modeling with sparse TCAM and Bayesian decoding*. Zenodo preprint. DOI: `10.5281/zenodo.18878295`

### Multi-domain native pre-training stage

- larger pre-training recipe
- multi-domain data mixture
- explicit activation sparsity reporting
- stronger evidence that native spike-based language-model training is feasible beyond toy data

Representative reference:

- Liu, T., Liu, Y., & Chen, W. (2026). *Scaling natively-trained spiking language models to multi-domain pre-training with 85% global activation sparsity*. SSRN preprint. DOI: `10.2139/ssrn.6427718`

### Dual-path 194M stage

- exponential-decay aggregation path
- spike-gated local attention path
- dynamic prior decoding head
- controlled comparison against GPT-2 baselines
- four independent main runs

This stage supplies the main controlled evidence for the unified paper.

### 0.8B scale-up stage

- separately distributed cleaned checkpoint
- public tokenizer
- public executable inference and smoke-training scripts
- artifact-based reproducibility package

This stage supplies scale-up evidence rather than a fully post-trained assistant model.

## Why The Public Package Centers On One Repository

The old `SymbolicLight DualPath` release skeleton and the current `SymbolicLight-0.8B-open` package tell overlapping parts of the same story.
Publishing both as equal public roots would create version ambiguity and reproducibility drift.

For that reason, the current repository takes the role of the **single public root**.
Historical 194M evidence is preserved through documentation and run registries rather than by keeping an older public skeleton as a second main package.
