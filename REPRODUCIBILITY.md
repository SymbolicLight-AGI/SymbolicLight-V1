# Reproducibility Guide

This GitHub repository supports inspection of the SymbolicLight V1 architecture, tokenizer, training loop, inference code, smoke-test training path, and historical 194M training narrative.
It does not include the 0.8B weight binary, manuscript files, raw training text, raw validation text, or source-level data manifest.

## What Is Public In This Repository

- Model implementation in `src/`
- Tokenizer files in `tokenizer/`
- Historical 194M run registry in `train_runs_194m.json`
- Narrative documentation in `docs/`
- Public evaluation script `src/eval_08.py`
- Public smoke-test training script `src/train_base.py`
- Lightweight smoke-test notes in `artifacts/`
- License files covering code, scripts, tokenizer assets, and public documentation

## What Is Distributed Separately

- Cleaned weights-only 0.8B checkpoint
- Manuscript PDF and submission files
- External archive metadata, if published

## What Is Not Public

- Raw training text
- Raw validation text
- Source-level dataset names
- Source identifiers and download URLs
- Source-level dataset manifest
- Internal audit records that map every paper table entry back to non-public data shards
- Any rights to redistribute non-public training or validation data

## What Can Be Verified From This Repository Alone

1. The public Python modules can be imported.
2. The tokenizer assets are present.
3. The public training script can complete a minimal smoke-test training loop with the built-in `smoke` dataset.
4. The code paths used for model construction, data loading, generation, and training are inspectable.
5. The manuscript-level split between the 194M controlled study and the 0.8B scale-up release is documented inside the package rather than scattered across separate legacy repository skeletons.

## What Requires Separately Distributed Weights

1. Loading the 0.8B checkpoint.
2. Running generation smoke tests against the 0.8B checkpoint.
3. Inspecting checkpoint-level metadata and hashes.
4. Verifying tokenizer/checkpoint compatibility with the released 0.8B weights.

## What Cannot Be Reproduced Publicly

1. Full pre-training from scratch with the paper's original corpus mixture.
2. Exact held-out PPL tables tied to non-public validation shards.
3. End-to-end regeneration of all paper figures and tables from public assets alone.
4. Public redistribution of the full historical DualPath checkpoint set, older raw eval outputs, or non-public ablation artifacts not included in this repository.

## Verified Command: Smoke-Test Training Loop

The following command was executed successfully in the release package on 2026-04-28 and can be rerun from this GitHub repository.

```powershell
cd <repo-root>
python -u src\train_base.py --dataset smoke --total_tokens 32 --batch_size 1 --grad_accum 1 --max_seq_len 16 --vocab_size 57344 --embed_dim 64 --n_layers 1 --n_heads 2 --head_dim 32 --intermediate_dim 128 --sparse_attn_window 16 --warmup_steps 1 --lr 1e-4 --save_every 1000 --keep_checkpoints 1 --save_dir artifacts\tmp_train_smoke --log_every 1 --num_workers 0 --no_fp16 --no_grad_checkpoint
```

Expected public outcome:

- Exit code `0`
- A two-step smoke run completes
- Training log is written to the console and can be redirected locally if desired
- The command creates an ephemeral `artifacts/tmp_train_smoke/` directory during local verification
- The ephemeral directory is intentionally not included in the release package

## Optional Command: Generation With Separately Downloaded Weights

After placing the separately distributed cleaned checkpoint at `weights/pytorch/latest.pt`, users may run:

```powershell
cd <repo-root>
python -u src\eval_08.py --checkpoint_path weights\pytorch\latest.pt --tokenizer_path tokenizer\sl_tokenizer.model --generate_only --max_new_tokens 4 --temperature 0.6 --top_k 20 --device cuda --allow_windows_cuda
```

## License and Data Boundary

The released code, scripts, tokenizer assets, and public documentation are licensed under Apache License, Version 2.0.
Separately distributed cleaned SymbolicLight V1 weights are intended to use Apache-2.0 unless their hosting page states otherwise.
The original training and validation data are not included, not redistributed, and not licensed through this repository.
Only aggregate domain categories, mixture proportions, preprocessing rules, and artifact-level verification metadata are public.
