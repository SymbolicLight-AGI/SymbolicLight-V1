# Artifact Notes

This directory contains lightweight public notes generated from the release package.
The 0.8B checkpoint binary and manuscript files are not stored in this GitHub repository.

## Files

- `train_smoke_test.log`: log from the minimal smoke-test training loop
- `BENCHMARK_08B.md`: conservative notes for local 0.8B release auditing with separately distributed weights

## Notes

- The smoke-test artifacts are not paper-quality benchmarks.
- They are included to show that the public training entry point runs end to end.
- Checkpoint-level generation and PPL checks require separately distributed weights.
- Temporary smoke-test checkpoints and structured local run outputs are intentionally not included in the release package.
  They are regenerated when users run the smoke-test training command in `REPRODUCIBILITY.md`.
