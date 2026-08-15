# Inference compatibility verification

The inference-only definitions were compared with the corresponding original
model definitions on an NVIDIA GeForce RTX 5090 using Python 3.12.3,
PyTorch 2.8.0+cu128, and CUDA 12.8.

| Check | 194M | 0.8B |
| --- | ---: | ---: |
| State-dict entries | 174 / 174 | 337 / 337 |
| Missing or extra entries | 0 | 0 |
| Full-forward maximum absolute logits difference | 0 | 0 |
| Cached-prefill maximum absolute logits difference | 0 | 0 |
| Cached next-token maximum absolute logits difference | 0 | 0 |
| Deterministic generated token sequence | Exact match | Exact match |
| Parameters unchanged after inference | Yes | Yes |
| BF16 and FP16 finite-logit smoke tests | Pass | Pass |
| End-to-end tokenizer and generation smoke test | Pass | Pass |

The 194M numerical comparison used a weights-only FP32 checkpoint. The 0.8B
GPU comparison used a deterministic BF16 mirror of the public FP32 tensors to
cover the intended 5090 inference path. Separately, the final FP32
inference-only checkpoints were compared with their source checkpoints tensor
by tensor: all names, shapes, dtypes, and values matched exactly.

This verification covers checkpoint loading and inference compatibility. It
does not reproduce training, ablations, held-out evaluation, or paper metrics.
