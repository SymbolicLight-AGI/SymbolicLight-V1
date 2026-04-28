#!/usr/bin/env python3
"""
SymbolicLight 0.8B evaluation script.

Features:
  1. Load a step_*.pt checkpoint
  2. Compute held-out CE / PPL with val.bin or a train-tail fallback
  3. Run generation quality tests on English and code prompts
  4. Report sparsity statistics

Examples:
  python eval_08.py --generate_only
  python eval_08.py --data_bin /path/to/data_bin --generate
  python eval_08.py --checkpoint_path ../weights/pytorch/latest.pt --generate_only
  python eval_08.py --data_bin /path/to/data_bin --json
"""
import argparse
import io
import json
import math
import os
import pickle
import struct
import sys
import time
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from model import SymbolicLightConfig, SymbolicLightModel


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_SL_TOKENIZER_PATH = str(PACKAGE_ROOT / "tokenizer" / "sl_tokenizer.model")

_StorageRef = namedtuple("_StorageRef", "dtype key location size")
_TensorRef = namedtuple("_TensorRef", "storage offset size stride requires_grad hooks")


def _rebuild_tensor_ref(storage, storage_offset, size, stride, requires_grad=False,
                        backward_hooks=None, metadata=None):
    return _TensorRef(
        storage=storage,
        offset=storage_offset,
        size=tuple(size),
        stride=tuple(stride),
        requires_grad=requires_grad,
        hooks=backward_hooks,
    )


def _rebuild_parameter_ref(data, requires_grad, backward_hooks):
    return data


class _CheckpointMetadataUnpickler(pickle.Unpickler):
    """Read PyTorch checkpoint metadata without materializing storages."""

    def persistent_load(self, pid):
        kind, storage_type, key, location, size = pid
        if kind != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {pid}")
        return _StorageRef(storage_type.__name__, key, location, size)

    def find_class(self, module, name):
        if module == "torch._utils" and name in {"_rebuild_tensor", "_rebuild_tensor_v2"}:
            return _rebuild_tensor_ref
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter_ref
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        return super().find_class(module, name)





DEFAULT_08B_CONFIG = dict(
    vocab_size=57344,
    embed_dim=1536,
    n_layers=22,
    n_heads=24,
    head_dim=64,
    intermediate_dim=6144,
    max_seq_len=512,
)

DEFAULT_CHECKPOINT = os.path.join(
    PACKAGE_ROOT, "weights", "pytorch", "latest.pt"
)





def parse_args():
    p = argparse.ArgumentParser(description="SymbolicLight 0.8B evaluation / PPL / generation test")

    
    p.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT,
                   help="Path to .pt checkpoint")
    p.add_argument("--tokenizer_path", type=str, default=_SL_TOKENIZER_PATH,
                   help="Path to SL tokenizer model")
    p.add_argument("--device", type=str, default="cpu",
                   help="cuda / cpu")
    p.add_argument("--allow_windows_cuda", action="store_true",
                   help="Allow experimental CUDA transfer on Windows")
    p.add_argument("--load_dtype", type=str, default="auto",
                   choices=["auto", "fp32", "fp16"],
                   help="Weight dtype during loading; auto uses fp16 on Windows")

    
    p.add_argument("--data_bin", type=str, default=None,
                   help="Directory containing val.bin/val.meta.json or train.bin/train.meta.json")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=128,
                   help="Maximum number of eval batches")
    p.add_argument("--seq_len", type=int, default=None,
                   help="Override sequence length; default reads from checkpoint config")
    p.add_argument("--allow_train_tail_fallback", action="store_true",
                   help="If val.bin is absent, evaluate on the tail of train.bin")
    p.add_argument("--tail_ratio", type=float, default=0.01,
                   help="When falling back to train.bin, reserve this tail fraction")
    p.add_argument("--tail_tokens_min", type=int, default=2_000_000,
                   help="Minimum number of tokens for pseudo-val region")

    
    p.add_argument("--generate", action="store_true",
                   help="Run generation quality tests after PPL evaluation")
    p.add_argument("--generate_only", action="store_true",
                   help="Skip PPL evaluation and run generation only")
    p.add_argument("--prompts", type=str, nargs="*", default=None,
                   help="Custom generation prompts appended after the built-in set")
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="Maximum number of generated tokens per prompt")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature")
    p.add_argument("--top_k", type=int, default=50,
                   help="Top-k sampling cutoff")
    p.add_argument("--top_p", type=float, default=0.9,
                   help="Top-p nucleus sampling threshold")
    p.add_argument("--repetition_penalty", type=float, default=1.1,
                   help="Repetition penalty (1.0 disables the penalty)")
    p.add_argument("--json", action="store_true",
                   help="Print final eval summary as JSON")
    return p.parse_args()





def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> dict:
    """Load a checkpoint while supporting both older and newer formats."""
    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    print(f"[Load] Loading checkpoint: {path}")
    ckpt = torch.load(str(path), map_location=device, weights_only=True, mmap=True)
    return ckpt


def load_checkpoint_metadata(checkpoint_path: str) -> dict:
    """Load checkpoint metadata only, keeping tensor storages as lightweight refs."""
    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    with zipfile.ZipFile(path) as archive:
        data_pkl = next(
            name for name in archive.namelist()
            if name.endswith("/data.pkl")
        )
        payload = archive.read(data_pkl)
    return _CheckpointMetadataUnpickler(io.BytesIO(payload)).load()


def build_config_from_checkpoint(ckpt: dict, args) -> Tuple[SymbolicLightConfig, int]:
    """Restore the model configuration from a checkpoint."""
    global_step = int(ckpt.get("global_step", ckpt.get("step", 0)))

    
    if "config" in ckpt:
        cfg_dict = ckpt["config"]
        if isinstance(cfg_dict, dict):
            valid_keys = {f.name for f in SymbolicLightConfig.__dataclass_fields__.values()}
            filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
            config = SymbolicLightConfig(**filtered)
        else:
            config = cfg_dict  
    else:
        
        print("[Load] [WARN] checkpoint has no config field; using the default 0.8B configuration")
        config = SymbolicLightConfig(**DEFAULT_08B_CONFIG)

    if args.seq_len is not None:
        config.max_seq_len = int(args.seq_len)

    return config, global_step


def prepare_model_state(ckpt: dict) -> dict:
    """Prepare a checkpoint state_dict for loading into SymbolicLightModel."""
    raw_state = ckpt.get("model", ckpt.get("model_state_dict", None))
    if raw_state is None:
        
        raw_state = {k: v for k, v in ckpt.items()
                     if isinstance(v, torch.Tensor)}

    
    cleaned = {}
    for k, v in raw_state.items():
        new_k = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned[new_k] = v

    
    cleaned = {k: v for k, v in cleaned.items() if "v_mem" not in k}
    return cleaned


def load_model_state(model: SymbolicLightModel, ckpt: dict, assign: bool = False) -> dict:
    """Load model weights with DDP and torch.compile key cleanup."""
    cleaned = prepare_model_state(ckpt)

    incompatible = model.load_state_dict(cleaned, strict=False, assign=assign)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def build_model_from_checkpoint(config: SymbolicLightConfig, ckpt: dict) -> Tuple[SymbolicLightModel, dict]:
    """Build the 0.8B model without duplicating parameter storage during load."""
    with torch.device("meta"):
        model = SymbolicLightModel(config)
    load_info = load_model_state(model, ckpt, assign=True)
    return model, load_info


def _dtype_from_storage(storage_dtype: str) -> torch.dtype:
    mapping = {
        "FloatStorage": torch.float32,
        "HalfStorage": torch.float16,
        "BFloat16Storage": torch.bfloat16,
        "DoubleStorage": torch.float64,
        "BoolStorage": torch.bool,
        "ByteStorage": torch.uint8,
        "CharStorage": torch.int8,
        "ShortStorage": torch.int16,
        "IntStorage": torch.int32,
        "LongStorage": torch.int64,
    }
    if storage_dtype not in mapping:
        raise TypeError(f"Unsupported checkpoint storage dtype: {storage_dtype}")
    return mapping[storage_dtype]


def _numpy_dtype_from_storage(storage_dtype: str):
    mapping = {
        "FloatStorage": np.float32,
        "HalfStorage": np.float16,
        "DoubleStorage": np.float64,
        "BoolStorage": np.bool_,
        "ByteStorage": np.uint8,
        "CharStorage": np.int8,
        "ShortStorage": np.int16,
        "IntStorage": np.int32,
        "LongStorage": np.int64,
    }
    if storage_dtype not in mapping:
        raise TypeError(f"Unsupported checkpoint storage dtype: {storage_dtype}")
    return mapping[storage_dtype]


def _checkpoint_zip_prefix(archive: zipfile.ZipFile) -> str:
    data_pkl = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
    return data_pkl.rsplit("/", 1)[0]


def _zip_stored_data_offset(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> int:
    archive.fp.seek(info.header_offset)
    header = archive.fp.read(30)
    if len(header) != 30 or header[:4] != b"PK\x03\x04":
        raise zipfile.BadZipFile(f"Invalid local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    return info.header_offset + 30 + name_len + extra_len


def _resolve_load_dtype(load_dtype: str) -> Optional[torch.dtype]:
    if load_dtype == "auto":
        return torch.float16 if os.name == "nt" else None
    if load_dtype == "fp16":
        return torch.float16
    return None


def _tensor_from_zip_ref(
    archive: zipfile.ZipFile,
    prefix: str,
    ref: _TensorRef,
    target_float_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    storage = ref.storage
    np_dtype = _numpy_dtype_from_storage(storage.dtype)
    info = archive.getinfo(f"{prefix}/data/{storage.key}")
    if info.compress_type == zipfile.ZIP_STORED:
        data_offset = _zip_stored_data_offset(archive, info)
        array = np.memmap(
            archive.filename,
            mode="r",
            dtype=np_dtype,
            offset=data_offset,
            shape=(storage.size,),
        )
    else:
        raw = archive.read(info.filename)
        array = np.frombuffer(raw, dtype=np_dtype).copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        tensor = torch.from_numpy(array)
    if ref.offset:
        tensor = tensor[ref.offset:]
    tensor = tensor.as_strided(ref.size, ref.stride)
    if target_float_dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(target_float_dtype)
    else:
        tensor = tensor.clone()
    return tensor


def _set_module_tensor(model: nn.Module, name: str, tensor: torch.Tensor) -> bool:
    module_name, _, leaf = name.rpartition(".")
    module = model.get_submodule(module_name) if module_name else model
    if leaf in module._parameters:
        old_param = module._parameters[leaf]
        requires_grad = True if old_param is None else old_param.requires_grad
        module._parameters[leaf] = nn.Parameter(tensor, requires_grad=requires_grad)
        return True
    if leaf in module._buffers:
        module._buffers[leaf] = tensor
        return True
    return False


def build_model_from_checkpoint_zip(
    config: SymbolicLightConfig,
    checkpoint_path: str,
    ckpt_metadata: dict,
    target_float_dtype: Optional[torch.dtype] = None,
) -> Tuple[SymbolicLightModel, dict]:
    """Load model weights directly from the checkpoint zip storages.

    This avoids a Windows PyTorch mmap failure observed with large CUDA-origin
    checkpoints where metadata loads correctly but tensor access segfaults.
    """
    raw_state = ckpt_metadata.get("model", ckpt_metadata.get("model_state_dict", None))
    if raw_state is None:
        raise KeyError("Checkpoint does not contain a model state_dict")

    cleaned = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in raw_state.items()
        if "v_mem" not in k
    }

    with torch.device("meta"):
        model = SymbolicLightModel(config)

    expected_keys = set(model.state_dict().keys())
    loaded_keys = set()
    unexpected_keys = []
    path = Path(checkpoint_path).resolve()
    with zipfile.ZipFile(path) as archive:
        prefix = _checkpoint_zip_prefix(archive)
        for name, ref in cleaned.items():
            if name not in expected_keys:
                unexpected_keys.append(name)
                continue
            tensor = _tensor_from_zip_ref(
                archive,
                prefix,
                ref,
                target_float_dtype=target_float_dtype,
            )
            if _set_module_tensor(model, name, tensor):
                loaded_keys.add(name)
            else:
                unexpected_keys.append(name)

    return model, {
        "missing_keys": sorted(expected_keys - loaded_keys),
        "unexpected_keys": sorted(unexpected_keys),
    }





def resolve_eval_region(data_bin_dir: Path, allow_train_tail_fallback: bool,
                        tail_ratio: float, tail_tokens_min: int):
    """Locate the evaluation region in val.bin or a fallback train tail."""
    val_bin = data_bin_dir / "val.bin"
    val_meta = data_bin_dir / "val.meta.json"
    if val_bin.exists() and val_meta.exists():
        with open(val_meta, encoding="utf-8") as f:
            meta = json.load(f)
        return val_bin, meta, 0, int(meta["total_tokens"]), "held-out val.bin"

    if not allow_train_tail_fallback:
        raise FileNotFoundError(
            "val.bin / val.meta.json not found. "
            "Provide a separate validation split or pass --allow_train_tail_fallback."
        )

    train_bin = data_bin_dir / "train.bin"
    train_meta = data_bin_dir / "train.meta.json"
    if not (train_bin.exists() and train_meta.exists()):
        raise FileNotFoundError(f"train.bin / train.meta.json does not exist: {data_bin_dir}")

    with open(train_meta, encoding="utf-8") as f:
        meta = json.load(f)
    total_tokens = int(meta["total_tokens"])
    tail_tokens = max(int(total_tokens * tail_ratio), int(tail_tokens_min))
    tail_tokens = min(tail_tokens, total_tokens)
    start_idx = max(0, total_tokens - tail_tokens)
    return train_bin, meta, start_idx, total_tokens, "pseudo-val tail of train.bin"


def run_ppl_eval(model, config, args, device):
    """Compute CE / PPL on a memmap dataset."""
    bin_path, meta, region_start, region_end, split_desc = resolve_eval_region(
        Path(args.data_bin),
        allow_train_tail_fallback=args.allow_train_tail_fallback,
        tail_ratio=args.tail_ratio,
        tail_tokens_min=args.tail_tokens_min,
    )

    dtype_str = meta.get("dtype", "uint16")
    if dtype_str != "uint16":
        raise ValueError(f"This evaluation path only supports uint16 memmap data, got {dtype_str}")

    seq_len = int(config.max_seq_len)
    usable_tokens = max(0, region_end - region_start)
    available_windows = max(0, (usable_tokens - 1) // seq_len)
    eval_batches = min(args.max_batches, available_windows // max(1, args.batch_size))
    if eval_batches <= 0:
        raise ValueError(
            f"Not enough usable validation windows: usable_tokens={usable_tokens}, "
            f"seq_len={seq_len}, batch_size={args.batch_size}"
        )

    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    autocast_enabled = device.type == "cuda"

    if not args.json:
        print(f"Eval split: {split_desc}")
        print(f"Eval region: [{region_start:,} - {region_end:,}] ({usable_tokens:,} tokens)")
        if "pseudo-val" in split_desc:
            print("[WARNING] Using the tail of train.bin as pseudo-validation; this is not a strict held-out split.")
        print(f"Seq len: {seq_len}")
        print(f"Batches: {eval_batches} x batch_size={args.batch_size}")

    total_loss = 0.0
    total_tokens = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx in range(eval_batches):
            batch_tokens = []
            base_offset = region_start + batch_idx * args.batch_size * seq_len
            for item_idx in range(args.batch_size):
                offset = base_offset + item_idx * seq_len
                if offset + seq_len + 1 > region_end:
                    break
                tokens = data[offset:offset + seq_len + 1].astype(np.int64)
                batch_tokens.append(tokens)

            if not batch_tokens:
                break

            batch = torch.tensor(np.array(batch_tokens), dtype=torch.long, device=device)
            batch = batch.clamp(0, config.vocab_size - 1)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(inputs)
            else:
                logits = model(inputs)

            loss_sum = F.cross_entropy(
                logits.float().reshape(-1, config.vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss_sum.item())
            total_tokens += int(targets.numel())

            if not args.json and (batch_idx == 0 or (batch_idx + 1) % 20 == 0):
                mean_ce = total_loss / max(total_tokens, 1)
                ppl = math.exp(min(mean_ce, 20.0))
                print(f"[Eval] batch {batch_idx+1}/{eval_batches} | "
                      f"CE={mean_ce:.4f} | PPL={ppl:.1f}")

    elapsed = time.time() - t_start
    mean_ce = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(mean_ce, 20.0))

    
    sparsity_stats = {}
    try:
        sparsity_stats = model.get_sparsity_stats()
    except Exception:
        pass

    return {
        "split": split_desc,
        "mean_ce": float(mean_ce),
        "ppl": float(ppl),
        "tokens": int(total_tokens),
        "batches": int(eval_batches),
        "elapsed_sec": round(elapsed, 2),
        "sparsity": sparsity_stats,
    }





DEFAULT_GENERATION_PROMPTS = [
    "Artificial intelligence can improve society by",
    "A good scientific explanation should",
    "Once upon a time, in a small village,",
    "def fibonacci(n):\n    ",
    "The Earth completes one orbit around the Sun in about",
    "Photosynthesis is the process by which",
    "In a world shaped by rapid scientific progress,",
    "The capital of France is",
]


def run_generation_test(model, tokenizer, args, device, global_step, config):
    """Run generation quality tests with cached incremental decoding."""
    prompts = list(DEFAULT_GENERATION_PROMPTS)
    if args.prompts:
        prompts.extend(args.prompts)

    print("\n" + "=" * 60)
    print(f" Generation Quality Test  (step {global_step})")
    print(f" temperature={args.temperature} | top_k={args.top_k} | "
          f"rep_penalty={args.repetition_penalty} | "
          f"max_tokens={args.max_new_tokens}")
    print("=" * 60)

    results = []
    for i, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        prompt_len = input_ids.size(1)

        t_start = time.time()
        
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        t_elapsed = time.time() - t_start

        generated_ids = output_ids[0].tolist()
        generated_text = tokenizer.decode(generated_ids)
        new_tokens = len(generated_ids) - prompt_len
        tok_per_sec = new_tokens / max(t_elapsed, 1e-6)

        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(prompts)}] Prompt: {prompt}")
        print(f"{'─'*60}")
        print(generated_text)
        print(f"  ({new_tokens} new tokens | {tok_per_sec:.1f} tok/s | {t_elapsed:.2f}s)")

        results.append({
            "prompt": prompt,
            "generated": generated_text,
            "new_tokens": new_tokens,
            "tok_per_sec": round(tok_per_sec, 1),
        })

    print(f"\n{'='*60}")
    print(f" Generation test complete: {len(results)} prompts")
    print(f"{'='*60}")
    return results





class TokenizerWrapper:
    """Small compatibility wrapper around the available tokenizer implementations."""
    def __init__(self, tokenizer_path: str):
        self.path = tokenizer_path
        try:
            from train_tokenizer import SLTokenizer
            self.tok = SLTokenizer(tokenizer_path)
            self._type = "sl"
            print(f"[Tokenizer] SLTokenizer loaded: {tokenizer_path}")
        except Exception:
            import sentencepiece as spm
            self.tok = spm.SentencePieceProcessor(model_file=tokenizer_path)
            self._type = "sp"
            print(f"[Tokenizer] SentencePiece loaded: {tokenizer_path}")

    def encode(self, text: str) -> list:
        if self._type == "sl":
            return self.tok.encode(text, add_bos=True, add_eos=False)
        return self.tok.encode(text)

    def decode(self, ids: list) -> str:
        if self._type == "sl":
            return self.tok.decode(ids)
        return self.tok.decode(ids)

    def eos_id(self) -> int:
        if self._type == "sl":
            return self.tok.sp.eos_id()
        return self.tok.eos_id()





def main():
    args = parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if os.name == "nt" and device.type == "cuda" and not args.allow_windows_cuda:
        print("[Device] Windows CUDA path is experimental for this checkpoint; using CPU for safe smoke tests.")
        print("[Device] Re-run with --allow_windows_cuda only if you want to test the CUDA transfer path.")
        device = torch.device("cpu")

    
    if os.name == "nt":
        print("[Load] Windows detected: using zip-storage checkpoint loader")
        target_float_dtype = _resolve_load_dtype(args.load_dtype)
        if target_float_dtype is not None:
            print(f"[Load] Loading floating-point weights as {target_float_dtype}")
        ckpt = load_checkpoint_metadata(args.checkpoint_path)
        config, global_step = build_config_from_checkpoint(ckpt, args)
        model, load_info = build_model_from_checkpoint_zip(
            config,
            args.checkpoint_path,
            ckpt,
            target_float_dtype=target_float_dtype,
        )
    else:
        ckpt = load_checkpoint(args.checkpoint_path, device="cpu")
        config, global_step = build_config_from_checkpoint(ckpt, args)
        model, load_info = build_model_from_checkpoint(config, ckpt)
    model.to(device)
    model.eval()

    
    del ckpt

    n_params = sum(p.numel() for p in model.parameters())

    if not args.json:
        print("=" * 60)
        print(" SymbolicLight 0.8B Evaluation")
        print("=" * 60)
        print(f"Checkpoint: {args.checkpoint_path}")
        print(f"Global step: {global_step}")
        print(f"Parameters: {n_params/1e6:.1f}M ({n_params/1e9:.3f}B)")
        print(f"Config: embed_dim={config.embed_dim}, n_layers={config.n_layers}, "
              f"n_heads={config.n_heads}, seq_len={config.max_seq_len}")
        print(f"Device: {device}")
        if load_info["missing_keys"]:
            print(f"[WARN] Missing keys: {load_info['missing_keys'][:5]}...")
        if load_info["unexpected_keys"]:
            print(f"[WARN] Unexpected keys: {load_info['unexpected_keys'][:5]}...")

    
    if args.generate_only:
        tokenizer = TokenizerWrapper(args.tokenizer_path)
        gen_results = run_generation_test(
            model, tokenizer, args, device, global_step, config
        )
        if args.json:
            summary = {
                "checkpoint": args.checkpoint_path,
                "global_step": global_step,
                "parameters": n_params,
                "generation_results": gen_results,
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    
    if not args.data_bin:
        raise ValueError(
            "PPL evaluation requires --data_bin. "
            "If you only want generation tests, pass --generate_only."
        )

    eval_result = run_ppl_eval(model, config, args, device)

    if not args.json:
        print("-" * 60)
        print(f"[Eval Done] mean_CE={eval_result['mean_ce']:.4f} | "
              f"PPL={eval_result['ppl']:.2f} | "
              f"tokens={eval_result['tokens']:,} | "
              f"time={eval_result['elapsed_sec']}s")
        if eval_result["sparsity"]:
            print("[Sparsity]")
            for k, v in eval_result["sparsity"].items():
                print(f"  {k}: {v*100:.1f}% silent")

    
    gen_results = None
    if args.generate:
        tokenizer = TokenizerWrapper(args.tokenizer_path)
        gen_results = run_generation_test(
            model, tokenizer, args, device, global_step, config
        )

    
    if args.json:
        summary = {
            "checkpoint": args.checkpoint_path,
            "global_step": global_step,
            "parameters": n_params,
            **eval_result,
        }
        if gen_results:
            summary["generation_results"] = gen_results
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
