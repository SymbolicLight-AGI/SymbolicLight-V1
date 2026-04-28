# SymbolicLight V1 0.8B Lightweight Benchmark

Date: 2026-04-28

This benchmark evaluates the released SymbolicLight V1 0.8B checkpoint in a lightweight, pretraining-oriented setting. It is not an instruction-following, MMLU, GPQA, coding-contest, or agent benchmark.

## Artifact

- Checkpoint: `weights/pytorch/latest.pt`
- Tokenizer: `tokenizer/sl_tokenizer.model`
- Global step: `186000`
- Parameters: `873,668,135`
- Device: `cuda`

## Held-Out PPL

This run used an internal held-out memmap shard.
The shard is not distributed with the public package, so this number should be read as a local release audit rather than a fully public benchmark.

Command pattern:

```powershell
python -u src\eval_08.py `
  --checkpoint_path weights\pytorch\latest.pt `
  --tokenizer_path tokenizer\sl_tokenizer.model `
  --data_bin <local-held-out-data-bin> `
  --batch_size 1 `
  --max_batches 128 `
  --device cuda `
  --allow_windows_cuda `
  --json
```

Result:

- Split: `held-out val.bin`
- Tokens evaluated: `65,536`
- Batches: `128`
- Mean CE: `3.1147`
- PPL: `22.53`
- Elapsed time: `45.43 s`
- Teacher-forced throughput: `~1,443 tokens/s`

Log: `artifacts/benchmark_08b_val128_ppl.log`

## Generation Smoke Benchmark

This run used the same internal held-out memmap shard for the lightweight PPL component.
The generation prompts are public, but the local PPL shard is not distributed.

Command pattern:

```powershell
python -u src\eval_08.py `
  --checkpoint_path weights\pytorch\latest.pt `
  --tokenizer_path tokenizer\sl_tokenizer.model `
  --data_bin <local-held-out-data-bin> `
  --batch_size 1 `
  --max_batches 16 `
  --device cuda `
  --allow_windows_cuda `
  --generate `
  --max_new_tokens 16 `
  --temperature 0.6 `
  --top_k 20 `
  --repetition_penalty 1.1 `
  --json
```

Result:

- Prompts: `8` English/code prompts
- New tokens: `16` per prompt
- Observed decoding speed: `12.9-17.0 tokens/s`
- Average decoding speed: `15.7 tokens/s`
- Lightweight PPL on the first `8,192` validation tokens: `22.61`

Representative outputs:

- `Artificial intelligence can improve society by providing new opportunities for the development of new technologies. The technology is now available to`
- `A good scientific explanation should be a good explanation of a phenomenon, but the explanation is not the only thing`
- `The capital of France is Paris. The capital of France is the capital of the country.`

The generation samples show basic continuation behavior but also factual weakness, e.g. the Earth-orbit prompt produced an incorrect continuation. This is consistent with the manuscript's framing of the 0.8B checkpoint as scale-up evidence rather than a quality claim.

Log: `artifacts/benchmark_08b_val16_gen16.log`

## Prompt-Level Sparsity Audit

Inputs: the same 8 English/code prompts used in the generation smoke benchmark.

Result:

- Encoder sparsity: `95.44%`
- Mean block sparsity: `93.77%`
- Min block sparsity: `67.21%`
- Max block sparsity: `96.40%`

Machine-readable output: `artifacts/benchmark_08b_sparsity_prompts.json`

## Interpretation

These results are suitable for an artifact/release note or a conservative paper appendix:

- The checkpoint loads successfully and runs on CUDA.
- The released tokenizer and checkpoint are compatible.
- The model can perform short English/code continuation.
- High activation sparsity is preserved in prompt-level inspection.
- The validation PPL measured on the local held-out shard is substantially weaker than the 194M controlled result and should not be presented as a dense-baseline quality comparison.

These results are not sufficient for claims about instruction following, factual reliability, coding ability, MMLU-style reasoning, or SOTA language-model quality.
