# Artifact Notes

This directory contains lightweight public verification artifacts generated from the released package.

## Files

- `generation_smoke_test.log`: log from the public 0.8B generation smoke test
- `train_smoke_test.log`: log from the minimal smoke-test training loop
- `checkpoint_metadata_summary.json`: lightweight summary of the released checkpoint and tokenizer

## Notes

- The smoke-test artifacts are not paper-quality benchmarks.
- They are included to show that the released package runs end to end on the public checkpoint and the public training entry point.
- Temporary smoke-test checkpoints and structured local run outputs are intentionally not included in the release package.
  They are regenerated when users run the smoke-test training command in `REPRODUCIBILITY.md`.
