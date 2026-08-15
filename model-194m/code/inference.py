#!/usr/bin/env python3
"""Minimal text generation for a released SymbolicLight V1 194M checkpoint."""

import argparse
from pathlib import Path

import torch

from model import SymbolicLightConfig, SymbolicLightModel
from tokenizer_runtime import SLTokenizer


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = PACKAGE_ROOT / "tokenizer" / "sl_tokenizer.model"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--no-adaptive-temperature", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def load_model(checkpoint_path: Path, device: torch.device) -> SymbolicLightModel:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=True, mmap=True
    )
    config_data = checkpoint.get("config", {})
    if isinstance(config_data, dict):
        valid = SymbolicLightConfig.__dataclass_fields__
        config = SymbolicLightConfig(**{k: v for k, v in config_data.items() if k in valid})
    else:
        config = config_data

    raw_state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    state = {
        key.replace("module.", "").replace("_orig_mod.", ""): value
        for key, value in raw_state.items()
        if isinstance(value, torch.Tensor) and "v_mem" not in key
    }
    with torch.device("meta"):
        model = SymbolicLightModel(config)
    incompatible = model.load_state_dict(state, strict=False, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device).eval()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer = SLTokenizer(args.tokenizer)
    model = load_model(args.checkpoint, device)

    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        adaptive_temperature=not args.no_adaptive_temperature,
    )
    generated_ids = output_ids[0, len(prompt_ids):].detach().cpu().tolist()
    print(tokenizer.decode(generated_ids))


if __name__ == "__main__":
    main()
