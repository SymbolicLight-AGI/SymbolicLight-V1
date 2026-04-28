# Reproducibility Guide

This package is intended to support artifact inspection, checkpoint loading, inference verification, smoke-test training, and inspection of the historical 194M training narrative.
It does not provide full end-to-end reproduction of the paper's pre-training results because the raw training text, raw validation text, and source-level data manifest are not public.

## What Is Public

- Model implementation in `src/`
- Tokenizer files in `tokenizer/`
- Released 0.8B checkpoint at `weights/pytorch/latest.pt`
- Historical 194M run registry in `train_runs_194m.json`
- Narrative documentation in `docs/`
- Public evaluation script `src/eval_08.py`
- Public smoke-test training script `src/train_base.py`
- Lightweight verification artifacts in `artifacts/`
- File checksums in `CHECKSUMS_SHA256.json`
- License files covering code, scripts, tokenizer assets, cleaned weights, and public documentation

## What Is Not Public

- Raw training text
- Raw validation text
- Source-level dataset names
- Source identifiers and download URLs
- Source-level dataset manifest
- Internal audit records that map every public table entry back to non-public data shards
- Any rights to redistribute non-public training or validation data

## What Can Be Verified From This Package

1. The public checkpoint can be loaded on Windows with the zip-storage loader.
2. The released tokenizer and checkpoint are mutually compatible.
3. The 0.8B checkpoint can run generation smoke tests through `src/eval_08.py`.
4. The public training script can complete a minimal smoke-test training loop with the built-in `smoke` dataset.
5. The code paths used for loading, generation, and training are executable in the released package.
6. The manuscript-level split between the 194M controlled study and the 0.8B scale-up release is documented inside the package rather than scattered across separate legacy repository skeletons.

## What Cannot Be Reproduced Publicly

1. Full pre-training from scratch with the paper's original corpus mixture.
2. Exact held-out PPL tables tied to non-public validation shards.
3. End-to-end regeneration of all paper figures and tables from public assets alone.
4. Public redistribution of the full historical DualPath checkpoint set, older raw eval outputs, or non-public ablation artifacts not included in this package.

## Verified Commands

The following commands were executed successfully in this package on 2026-04-28.

### 0.8B Generation Smoke Test

```powershell
cd <repo-root>
python -u src\eval_08.py --checkpoint_path weights\pytorch\latest.pt --tokenizer_path tokenizer\sl_tokenizer.model --generate_only --max_new_tokens 4 --temperature 0.6 --top_k 20 --device cuda --allow_windows_cuda
```

Expected public outcome:

- Exit code `0`
- Checkpoint loads successfully
- Eight default English prompts run to completion
- Log is written in [artifacts/generation_smoke_test.log](D:/SNN/Open08B/SymbolicLight-0.8B-open/artifacts/generation_smoke_test.log)

### Smoke-Test Training Loop

```powershell
cd <repo-root>
python -u src\train_base.py --dataset smoke --total_tokens 32 --batch_size 1 --grad_accum 1 --max_seq_len 16 --vocab_size 57344 --embed_dim 64 --n_layers 1 --n_heads 2 --head_dim 32 --intermediate_dim 128 --sparse_attn_window 16 --warmup_steps 1 --lr 1e-4 --save_every 1000 --keep_checkpoints 1 --save_dir artifacts\tmp_train_smoke --log_every 1 --num_workers 0 --no_fp16 --no_grad_checkpoint
```

Expected public outcome:

- Exit code `0`
- A two-step smoke run completes
- Training log is written in [artifacts/train_smoke_test.log](D:/SNN/Open08B/SymbolicLight-0.8B-open/artifacts/train_smoke_test.log)
- The command creates an ephemeral `artifacts/tmp_train_smoke/` directory during local verification.
  That directory is intentionally not included in the release package.

## License and Data Boundary

The released code, scripts, tokenizer assets, cleaned weights-only checkpoint, and public documentation are licensed under Apache License, Version 2.0.
The original training and validation data are not included, not redistributed, and not licensed through this repository.
Only aggregate domain categories, mixture proportions, preprocessing rules, and artifact-level verification metadata are public.

## Review Guidance

For manuscript review, the public package should be interpreted as an artifact-based reproducibility release with a unified historical narrative.
It supports inspection of the architecture, training loop, tokenizer, released checkpoint, public verification commands, and the documented 194M-to-0.8B project lineage.
It should not be interpreted as a complete public reconstruction of the private pre-training corpus or all paper metrics.
