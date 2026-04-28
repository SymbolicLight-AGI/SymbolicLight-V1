# 194M Training Story

This note preserves the main 194M experimental narrative that was previously spread across older DualPath release materials.

## Role Of The 194M Model

The 194M SymbolicLight V1 model is the **main controlled experimental anchor** for the paper.
It is the part of the project where the strongest direct comparison against parameter-matched dense GPT-2 baselines was carried out.

## What Was Established At 194M

- Stable native pre-training at high activation sparsity
- Four independent main runs with tightly grouped held-out validation perplexity
- Controlled comparison against GPT-2 124M and GPT-2 201M style dense baselines
- Mechanism ablations that separate temporal spike dynamics from simple deterministic sparsity

## Main 194M Runs

The four main 194M runs are recorded in [train_runs_194m.json](../train_runs_194m.json).

In summary:

- `snn_auxce_s123`
- `snn_auxce_s456`
- `snn_noauxce_s42`
- `snn_noauxce_s123`

These runs form the basis of the main validation-PPL range reported in the manuscript.

## Historical DualPath Release Claims Preserved Here

Older public README material emphasized:

- 194M parameters
- about 3B bilingual training tokens
- four independent seeds
- comparison against GPT-2 124M and GPT-2 201M baselines
- ablation checkpoints for no-attention, decay-only, static-prior, and top-k-mask variants

Those claims are not discarded.
Instead, they are absorbed into this repository as historical experiment documentation, while the public reproducibility boundary remains governed by the current package files and current paper text.

## Why The Full 194M Historical Package Is Not Reconstructed As-Is

The older DualPath-style public skeleton previously referred to a legacy layout with:

- `code/`
- `eval_scripts/`
- `eval_results/`
- a larger multi-checkpoint release layout

These names describe the historical DualPath release skeleton, not the current SymbolicLight V1 public package.
The current repository does not reconstruct that exact layout because doing so would reintroduce public-facing inconsistencies with the current manuscript and current release scope.

Instead, this repository preserves the 194M story through:

- the unified manuscript
- historical run registry
- public reproducibility notes
- public code paths that remain executable

## How To Read 194M vs 0.8B

- Read **194M** as the main controlled evidence for model quality under sparsity.
- Read **0.8B** as scale-up evidence for native pre-training feasibility.

That split is intentional and is the recommended interpretation for manuscript reviewers and external readers.
