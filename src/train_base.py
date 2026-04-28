#!/usr/bin/env python3
"""
SymbolicLight V1 pre-training entry point.

Examples:
  python train_base.py --dry_run
  python train_base.py --batch_size 56 --total_tokens 3000000000
  torchrun --nproc_per_node=4 train_base.py --batch_size 14 --total_tokens 3000000000
  torchrun --nproc_per_node=4 train_base.py --resume --batch_size 14 --total_tokens 3000000000
"""




import os
import subprocess


os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')



if os.environ.get("SYMBOLICLIGHT_INSECURE_TLS", "").lower() in {"1", "true", "yes"}:
    print("[Net] [WARN] SYMBOLICLIGHT_INSECURE_TLS is ignored for security.")
    print("[Net] [WARN] Fix certificates or proxy settings instead of weakening TLS verification.")
if False:  
    print("[Net] [WARN] ⚠️  TLS certificate verification WEAKENED for HF downloads")

os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '60')


if os.path.exists('/etc/network_turbo'):
    try:
        result = subprocess.run(
            'bash -c "source /etc/network_turbo && env | grep -i proxy"',
            shell=True, capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if '=' in line:
                var, value = line.split('=', 1)
                os.environ[var] = value
        if os.environ.get('http_proxy') or os.environ.get('https_proxy'):
            print("[Net] [OK] AutoDL network_turbo proxy loaded")
    except Exception:
        pass



_USE_MODELSCOPE = False
_has_proxy = bool(os.environ.get('http_proxy') or os.environ.get('https_proxy'))
if _has_proxy:
    os.environ.pop('HF_ENDPOINT', None)
    print("[Net] Proxy detected; using direct endpoint configuration")
else:
    print("[Net] No proxy detected. Set HF_ENDPOINT manually if your environment requires a mirror.")

import sys
import json
import time
import math
import random
import inspect
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime as dt


sys.path.insert(0, str(Path(__file__).parent))
from model import SymbolicLightConfig, SymbolicLightModel
from data_pipeline import (
    DEFAULT_CURRICULUM_PHASE_1_RATIO,
    DEFAULT_CURRICULUM_PHASE_2_RATIO,
    DEFAULT_CURRICULUM_PRESET,
    MemmapDataset,
    StreamingParquetDataset,
    format_source_histogram,
    get_curriculum_phase,
    resolve_tokenizer_path,
)


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_SL_TOKENIZER_PATH = str(_PACKAGE_ROOT / "tokenizer" / "sl_tokenizer.model")
if not os.path.exists(_SL_TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer model not found: {_SL_TOKENIZER_PATH}")
from train_tokenizer import SLTokenizer
print(f"[Tokenizer] [OK] SL tokenizer: {_SL_TOKENIZER_PATH}")





def setup_distributed():
    """Initialize distributed training when launched via torchrun."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            backend = os.environ.get('DIST_BACKEND', 'nccl')
        else:
            device = torch.device('cpu')
            backend = 'gloo'
        timeout = dt.timedelta(minutes=30)

        try:
            if device.type == "cuda":
                dist.init_process_group(backend, timeout=timeout, device_id=device)
            else:
                dist.init_process_group(backend, timeout=timeout)
        except Exception as e:
            if backend == 'nccl':
                print(f"[DDP] NCCL init failed ({e}), falling back to gloo...")
                dist.init_process_group('gloo', timeout=timeout)
            else:
                raise

        dist.barrier()
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0





DEFAULT_DATA_DIR = "./data/private_corpus"


def build_data_recipe(data_dir):
    """Build a domain-level recipe without exposing source-level dataset names."""
    return [
        {"name": "reference-web",       "local_dir": os.path.join(data_dir, "reference-web"),       "split": "train", "weight": 0.25, "text_key": "text"},
        {"name": "math-web",            "local_dir": os.path.join(data_dir, "math-web"),            "split": "train", "weight": 0.20, "text_key": "text"},
        {"name": "code-text",           "local_dir": os.path.join(data_dir, "code-text"),           "split": "train", "weight": 0.15, "text_key": "text"},
        {"name": "general-web",         "local_dir": os.path.join(data_dir, "general-web"),         "split": "train", "weight": 0.15, "text_key": "text"},
        {"name": "academic-educational","local_dir": os.path.join(data_dir, "academic-educational"),"split": "train", "weight": 0.10, "text_key": "text"},
        {"name": "open-educational",    "local_dir": os.path.join(data_dir, "open-educational"),    "split": "train", "weight": 0.08, "text_key": "text"},
        {"name": "synthetic-narrative", "local_dir": os.path.join(data_dir, "synthetic-narrative"), "split": "train", "weight": 0.05, "text_key": "text"},
        {"name": "translation",         "local_dir": os.path.join(data_dir, "translation"),         "split": "train", "weight": 0.02, "text_key": "translation"},
    ]





class SmokeTestStreamingDataset(IterableDataset):
    """Small synthetic stream used only to test the training loop."""
    def __init__(self, seq_len=512, vocab_size=57344):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.enc = SLTokenizer(_SL_TOKENIZER_PATH)

    def __iter__(self):
        examples = [
            "SymbolicLight uses sparse spike-gated computation for language modeling.",
            "This public smoke-test stream is not part of the reported training data.",
            "Use your own legally available corpus under the aggregate domain recipe.",
        ]
        token_buffer = []
        while True:
            text = examples[len(token_buffer) % len(examples)]
            tokens = self.enc.encode(text, add_bos=False)
            token_buffer.extend(tokens)
            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[:self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long).clamp(0, self.vocab_size - 1)
                y = torch.tensor(chunk[1:], dtype=torch.long).clamp(0, self.vocab_size - 1)
                yield x, y





def get_lr(step, warmup_steps, total_steps, max_lr, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))



def realign_data_iterator(dataloader, skip_target, rank, label="Resume", report_every=1000):
    """Replay the deterministic streaming pipeline until the saved batch offset."""
    data_iter = iter(dataloader)
    if skip_target <= 0:
        return data_iter, 0

    if is_main_process(rank):
        print(f"[{label}] Realigning data stream by skipping {skip_target} batches...")

    skip_count = 0
    while skip_count < skip_target:
        try:
            next(data_iter)
            skip_count += 1
        except StopIteration:
            data_iter = iter(dataloader)
            continue

        if report_every > 0 and skip_count % report_every == 0 and is_main_process(rank):
            print(f"[{label}] Skipped {skip_count}/{skip_target} batches...")

    if is_main_process(rank):
        print(f"[{label}] Data stream aligned after skipping {skip_count} batches")

    return data_iter, skip_count


def build_mixed_dataset(args, rank, world_size, vocab_size, seed_offset=0):
    tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
    if args.data_bin:
        return MemmapDataset(
            data_bin_dir=args.data_bin,
            seq_len=args.max_seq_len,
            rank=rank,
            world_size=world_size,
            model_vocab_size=vocab_size,
            seed_offset=seed_offset,
            strict_no_repeat=not args.allow_source_restarts,
        )
    return StreamingParquetDataset(
        data_dir=args.data_dir,
        tokenizer_path=tokenizer_path,
        seq_len=args.max_seq_len,
        rank=rank,
        world_size=world_size,
        model_vocab_size=vocab_size,
        seed_offset=seed_offset,
        max_oversample=args.max_oversample,
        strict_no_repeat=not args.allow_source_restarts,
    )


def maybe_switch_curriculum_phase(args, dataset, tokens_seen, current_phase, rank):
    phase, recipe = get_curriculum_phase(
        tokens_seen,
        args.total_tokens,
        phase1_ratio=args.curriculum_phase1_ratio,
        phase2_ratio=args.curriculum_phase2_ratio,
        preset=args.curriculum_preset,
    )
    if phase != current_phase:
        dataset.set_phase(recipe)
        current_phase = phase
        if is_main_process(rank):
            print(f"\n[Curriculum] Entering Phase {phase}/3 at {tokens_seen / 1e9:.2f}B tokens")
            if hasattr(dataset, "get_phase_summary"):
                phase_summary = dataset.get_phase_summary()
                active_sources = phase_summary.get("active_sources", {})
                unresolved_sources = phase_summary.get("unresolved_sources", [])
                if phase_summary.get("uses_legacy_spans"):
                    print("  NOTE: current data_bin has no explicit source_spans; using legacy source_stats spans")
                if active_sources:
                    print(f"  Active sources: {format_source_histogram(active_sources)}")
                if unresolved_sources:
                    print(f"  Unresolved sources: {', '.join(unresolved_sources)}")
            for source in recipe:
                print(f"  - {source['name']}: {source['weight'] * 100:.0f}%")
    return current_phase


def merge_source_histograms(histograms):
    merged = {}
    for histogram in histograms:
        if not histogram:
            continue
        for source_id, count in histogram.items():
            merged[source_id] = merged.get(source_id, 0) + int(count)
    return merged


def merge_source_sampling_stats(stats_list):
    merged_sources = {}
    for snapshot in stats_list:
        if not snapshot:
            continue
        for source_id, entry in snapshot.get("sources", {}).items():
            merged = merged_sources.setdefault(
                source_id,
                {
                    "mode": entry.get("mode", snapshot.get("mode", "unknown")),
                    "budget_is_exact": bool(entry.get("budget_is_exact", False)),
                    "replicated": False,
                    "sampled_train_tokens": 0,
                    "sampled_windows": 0,
                    "completed_files": 0,
                    "active": False,
                    "_all_exhausted": True,
                    "_budget_values": [],
                    "_window_values": [],
                    "_file_values": [],
                    "_remaining_window_values": [],
                },
            )
            replicated = bool(entry.get("replicated", False))
            merged["replicated"] = merged["replicated"] or replicated
            merged["budget_is_exact"] = merged["budget_is_exact"] or bool(entry.get("budget_is_exact", False))
            merged["sampled_train_tokens"] += int(entry.get("sampled_train_tokens", 0))
            merged["sampled_windows"] += int(entry.get("sampled_windows", 0))
            merged["completed_files"] += int(entry.get("completed_files", 0))
            merged["active"] = merged["active"] or bool(entry.get("active", False))
            merged["_all_exhausted"] = merged["_all_exhausted"] and bool(entry.get("exhausted", False))

            budget = int(entry.get("unique_token_budget", 0) or 0)
            total_windows = int(entry.get("total_windows", 0) or 0)
            total_files = int(entry.get("total_files", 0) or 0)
            if budget > 0:
                merged["_budget_values"].append((budget, replicated))
            if total_windows > 0:
                merged["_window_values"].append((total_windows, replicated))
            remaining_windows = int(entry.get("remaining_windows", 0) or 0)
            if remaining_windows > 0:
                merged["_remaining_window_values"].append((remaining_windows, replicated))
            if total_files > 0:
                merged["_file_values"].append((total_files, replicated))

    for source_id, entry in merged_sources.items():
        budget_values = entry.pop("_budget_values")
        window_values = entry.pop("_window_values")
        file_values = entry.pop("_file_values")
        remaining_window_values = entry.pop("_remaining_window_values")

        if budget_values:
            entry["unique_token_budget"] = (
                max(value for value, _ in budget_values)
                if entry["replicated"]
                else sum(value for value, _ in budget_values)
            )
        else:
            entry["unique_token_budget"] = 0

        if window_values:
            entry["total_windows"] = (
                max(value for value, _ in window_values)
                if entry["replicated"]
                else sum(value for value, _ in window_values)
            )
        else:
            entry["total_windows"] = 0

        if remaining_window_values:
            entry["remaining_windows"] = (
                max(value for value, _ in remaining_window_values)
                if entry["replicated"]
                else sum(value for value, _ in remaining_window_values)
            )
        else:
            entry["remaining_windows"] = 0

        if file_values:
            entry["total_files"] = (
                max(value for value, _ in file_values)
                if entry["replicated"]
                else sum(value for value, _ in file_values)
            )
        else:
            entry["total_files"] = 0

        entry["exhausted"] = entry.pop("_all_exhausted")
        budget = entry.get("unique_token_budget", 0)
        total_windows = entry.get("total_windows", 0)
        total_files = entry.get("total_files", 0)
        entry["epoch"] = (
            float(entry["sampled_train_tokens"]) / float(budget)
            if budget > 0
            else None
        )
        if total_windows > 0:
            entry["coverage"] = float(entry["sampled_windows"]) / float(total_windows)
        elif total_files > 0:
            entry["coverage"] = float(entry["completed_files"]) / float(total_files)
        else:
            entry["coverage"] = None

    return {"sources": merged_sources}


DATA_STATE_BUNDLE_KIND = "symboliclight_ranked_data_state"


def collect_checkpoint_data_state(dataset, *, use_direct_dataset, is_ddp, rank, world_size):
    if not use_direct_dataset or dataset is None or not hasattr(dataset, "state_dict"):
        return None

    local_state = dataset.state_dict()
    if not is_ddp:
        return local_state

    gathered_states = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_states, local_state)
    if not is_main_process(rank):
        return None

    return {
        "__kind__": DATA_STATE_BUNDLE_KIND,
        "__version__": 1,
        "world_size": int(world_size),
        "per_rank": {
            str(rank_id): gathered_states[rank_id]
            for rank_id in range(world_size)
            if gathered_states[rank_id] is not None
        },
    }


def resolve_rank_checkpoint_data_state(data_state, *, rank, world_size):
    if not data_state:
        return None, None

    if isinstance(data_state, dict) and data_state.get("__kind__") == DATA_STATE_BUNDLE_KIND:
        per_rank = data_state.get("per_rank", {})
        selected = per_rank.get(str(rank))
        stored_world_size = int(data_state.get("world_size", len(per_rank) or 1))

        warning = None
        if stored_world_size != int(world_size):
            warning = (
                f"checkpoint data_state was saved with world_size={stored_world_size}, "
                f"current world_size={world_size}; attempting rank-wise restore"
            )

        if selected is None:
            available = ", ".join(sorted(str(key) for key in per_rank)) or "none"
            return None, (
                f"checkpoint has no sampler state for rank {rank} "
                f"(available ranks: {available}); skipping dataset restore"
            )
        return selected, warning

    if int(world_size) > 1:
        return None, (
            "checkpoint only contains a single shared data_state. "
            "Skipping sampler restore under DDP to avoid cross-rank data duplication."
        )
    return data_state, None


def restore_checkpoint_data_state(dataset, data_state, *, rank, world_size, label="Resume"):
    if dataset is None or not hasattr(dataset, "load_state_dict") or not data_state:
        return False

    selected_state, warning = resolve_rank_checkpoint_data_state(
        data_state,
        rank=rank,
        world_size=world_size,
    )
    if warning and is_main_process(rank):
        print(f"[{label}] WARNING: {warning}")

    if selected_state is None:
        return False

    try:
        dataset.load_state_dict(selected_state)
        if is_main_process(rank):
            print(f"[{label}] Data sampler state restored")
        return True
    except Exception as exc:
        if is_main_process(rank):
            print(f"[{label}] WARNING: failed to restore data sampler state ({exc})")
        return False


def summarize_source_sampling_stats(snapshot, *, warn_threshold=0.80, top_k=4):
    if not snapshot:
        return "", [], {}

    sources = snapshot.get("sources", {})
    if not sources:
        return "", [], {}

    ranked = []
    compact = {}
    warnings = []

    def _fmt_metric(value):
        if value is None:
            return "n/a"
        return f"{value:.3f}" if abs(float(value)) < 0.01 else f"{value:.2f}"

    for source_id, entry in sources.items():
        epoch = entry.get("epoch")
        coverage = entry.get("coverage")
        mode = entry.get("mode", "unknown")
        sampled_train_tokens = int(entry.get("sampled_train_tokens", 0) or 0)
        score = epoch if epoch is not None else (coverage if coverage is not None else 0.0)
        if sampled_train_tokens <= 0 and (coverage is None or coverage <= 0):
            continue
        if mode == "memmap_exact":
            if epoch is None:
                continue
            label = f"{source_id}:{_fmt_metric(epoch)}ep"
            if coverage is not None:
                label += f"/{coverage * 100:.0f}%cov"
        else:
            observed = epoch if epoch is not None else 0.0
            label = f"{source_id}:{_fmt_metric(observed)}obs"
            if coverage is not None:
                label += f"/{coverage * 100:.0f}%files"
        ranked.append((score, source_id, label))
        compact[source_id] = {
            "mode": mode,
            "epoch": epoch,
            "coverage": coverage,
            "sampled_train_tokens": sampled_train_tokens,
            "unique_token_budget": entry.get("unique_token_budget", 0),
            "sampled_windows": entry.get("sampled_windows", 0),
            "total_windows": entry.get("total_windows", 0),
            "remaining_windows": entry.get("remaining_windows", 0),
            "completed_files": entry.get("completed_files", 0),
            "total_files": entry.get("total_files", 0),
            "replicated": entry.get("replicated", False),
            "active": entry.get("active", False),
            "exhausted": entry.get("exhausted", False),
        }

        trigger_value = epoch if entry.get("budget_is_exact", False) else coverage
        if warn_threshold > 0 and trigger_value is not None and trigger_value >= warn_threshold:
            if entry.get("budget_is_exact", False):
                warnings.append(f"{source_id}:{_fmt_metric(epoch)}ep")
            else:
                warnings.append(f"{source_id}:{coverage * 100:.0f}%files")

    ranked.sort(key=lambda item: (-item[0], item[1]))
    summary = ", ".join(label for _, _, label in ranked[: max(1, int(top_k))])
    warnings = sorted(set(warnings))
    return summary, warnings[: max(1, int(top_k))], compact





def parse_args():
    p = argparse.ArgumentParser(description="SymbolicLight 0.8B Trainer (DDP + Auto Aux CE)")
    p.add_argument("--data_bin", type=str, default=None,
                   help="Pretokenized memmap directory (train.bin + train.meta.json)")
    p.add_argument("--tokenizer_path", type=str, default=_SL_TOKENIZER_PATH,
                   help="SentencePiece tokenizer path for parquet/memmap pipeline")
    p.add_argument("--curriculum_phase1_ratio", type=float, default=DEFAULT_CURRICULUM_PHASE_1_RATIO,
                   help="Fraction of total tokens reserved for phase 1 before phase 2")
    p.add_argument("--curriculum_phase2_ratio", type=float, default=DEFAULT_CURRICULUM_PHASE_2_RATIO,
                   help="Fraction of total tokens reserved for phase 1+2 before phase 3")
    p.add_argument("--curriculum_preset", type=str, default=DEFAULT_CURRICULUM_PRESET,
                   choices=["default", "30b08", "30b08_30b", "50b08"],
                   help="Curriculum source-weight preset")
    p.add_argument("--max_oversample", type=float, default=5.0,
                   help="Cap phase source weight to natural source share * max_oversample")
    p.add_argument("--allow_source_restarts", action="store_true",
                   help="Allow exhausted sources/windows to restart and repeat")
    p.add_argument("--source_epoch_warn", type=float, default=0.80,
                   help="Warn when a source epoch/coverage reaches this threshold")

    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                   help="Root directory containing parquet sources")
    p.add_argument("--dataset", type=str, default="mixed",
                   choices=["mixed", "smoke"],
                   help="Dataset mode: mixed curriculum sampler or synthetic smoke test")
    p.add_argument("--total_tokens", type=int, default=3_000_000_000,
                   help="Total training tokens")

    p.add_argument("--vocab_size", type=int, default=57344,
                   help="Vocabulary size")
    p.add_argument("--embed_dim", type=int, default=1536)
    p.add_argument("--n_layers", type=int, default=22)
    p.add_argument("--n_heads", type=int, default=24)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--intermediate_dim", type=int, default=6144)
    p.add_argument("--max_seq_len", type=int, default=512,
                   help="Sequence length")

    p.add_argument("--batch_size", type=int, default=13,
                   help="Per-device batch size (lower it to avoid OOM while using grad_accum to preserve the effective batch)")
    p.add_argument("--grad_accum", type=int, default=2,
                   help="Gradient accumulation steps (with batch_size=13 this helps preserve the effective batch size)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=2000,
                   help="Warmup steps")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)


    p.add_argument("--fp16", dest="fp16", action="store_true",
                   help="Enable mixed precision training (default: on)")
    p.add_argument("--no_fp16", dest="fp16", action="store_false",
                   help="Disable mixed precision and run in FP32 for debugging")
    p.add_argument("--grad_checkpoint", dest="grad_checkpoint", action="store_true",
                   help="Enable activation checkpointing (default: on)")
    p.add_argument("--no_grad_checkpoint", dest="grad_checkpoint", action="store_false",
                   help="Disable activation checkpointing")
    p.set_defaults(fp16=True, grad_checkpoint=True)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. Streaming parquet on multi-GPU is safer with 0.")

    p.add_argument("--save_dir", type=str, default="checkpoints_0p8b")
    p.add_argument("--save_every", type=int, default=2000,
                   help="Save a checkpoint every N steps")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--keep_checkpoints", type=int, default=3,
                   help="Number of recent checkpoints to keep")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed")

    p.add_argument("--sparse_attn_window", type=int, default=512,
                   help="Sparse attention sliding window size (default: 512, covers full seq_len)")
    p.add_argument("--disable_sparse_attn", action="store_true",
                   help="Disable Sparse Local Attention (Decay-Only / No-Attn ablation)")
    p.add_argument("--disable_dynamic_prior", action="store_true",
                   help="Disable Dynamic Bayesian Prior (Static Prior ablation)")
    p.add_argument("--use_topk_mask", action="store_true",
                   help="[W1 Ablation] Replace LIF spike gating with a fixed top-k mask")
    p.add_argument("--topk_sparsity", type=float, default=0.89,
                   help="[W1 Ablation] Target sparsity for top-k mask")

    p.add_argument("--dry_run", action="store_true",
                   help="Run a short smoke test")

    return p.parse_args()





def train(args):
    
    print(f"[DEBUG] Rank {os.environ.get('RANK', '?')}: entering train()", flush=True)
    rank, local_rank, world_size = setup_distributed()
    is_ddp = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Rank {rank}: DDP init done, device={device}", flush=True)

    
    if args.dry_run:
        args.total_tokens = 200 * args.batch_size * args.max_seq_len * world_size
        args.save_every = 100
        args.log_every = 10
        if is_main_process(rank):
            print("\n" + "!" * 60)
            print("  DRY RUN MODE - validating the streaming data pipeline and model")
            print("!" * 60)

    if args.embed_dim != args.n_heads * args.head_dim:
        raise ValueError(
            f"embed_dim ({args.embed_dim}) must equal n_heads * head_dim "
            f"({args.n_heads} * {args.head_dim} = {args.n_heads * args.head_dim})"
        )

    if is_main_process(rank):
        print(f"\n{'=' * 60}")
        print(f" SymbolicLight V1 Pre-Training")
        print(f"{'=' * 60}")
        print(f"Device: {device}")
        print(f"World size: {world_size} GPU(s)")
        n_params_est = (
            args.vocab_size * args.embed_dim +
            args.n_layers * (
                3 * args.embed_dim * args.embed_dim +
                args.embed_dim * args.embed_dim +
                2 * args.embed_dim * args.intermediate_dim
            )
        )
        print(
            f"Model profile: ~{n_params_est / 1e6:.1f}M params | d_model={args.embed_dim}, layers={args.n_layers}, "
            f"heads={args.n_heads}, head_dim={args.head_dim}, ffn={args.intermediate_dim}"
        )
        if args.disable_sparse_attn:
            print(f"ABLATION: Sparse Attention DISABLED (Decay-Only mode)")
        if args.disable_dynamic_prior:
            print(f"ABLATION: Dynamic Prior DISABLED (Static Prior mode)")
        if getattr(args, 'use_topk_mask', False):
            print(f"ABLATION: Top-K Mask ENABLED (sparsity={args.topk_sparsity:.0%}, replacing LIF)")
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(local_rank)
            gpu_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    
    args._initial_lr = args.lr

    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if is_main_process(rank):
        print(f"Seed: {args.seed}")

    
    config = SymbolicLightConfig(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        intermediate_dim=args.intermediate_dim,
        max_seq_len=args.max_seq_len,
        enable_stdp=False,
        enable_sparse_attn=not args.disable_sparse_attn,
        sparse_attn_window=getattr(args, 'sparse_attn_window', 512),
        enable_dynamic_prior=not args.disable_dynamic_prior,
        use_topk_mask=getattr(args, 'use_topk_mask', False),
        topk_sparsity=getattr(args, 'topk_sparsity', 0.89),
    )
    if is_main_process(rank):
        print(f"[DEBUG] Creating model...", flush=True)
    model = SymbolicLightModel(config).to(device)
    
    
    if is_main_process(rank):
        print(f"[DEBUG] Model created and moved to {device}", flush=True)

    
    if is_ddp:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False)
        else:
            model = DDP(model, find_unused_parameters=False)
        if is_main_process(rank):
            print(f"[DDP] DistributedDataParallel enabled on {world_size} GPUs")

    
    if is_main_process(rank):
        print(f"[DEBUG] Creating dataset...", flush=True)
    dataset = None
    dataloader = None
    if args.dataset == "smoke":
        dataset = SmokeTestStreamingDataset(seq_len=args.max_seq_len, vocab_size=args.vocab_size)
        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.num_workers > 0,
        )
        if args.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = 2
        dataloader = DataLoader(**dataloader_kwargs)
    if is_main_process(rank):
        print(f"[DEBUG] Dataset handle prepared", flush=True)

    
    tokens_per_step = args.batch_size * args.max_seq_len * args.grad_accum * world_size
    total_steps = args.total_tokens // tokens_per_step

    if is_main_process(rank):
        print(f"\n[Training Plan]")
        print(f"  Total tokens:       {args.total_tokens / 1e9:.1f}B")
        print(f"  Tokens per step:    {tokens_per_step:,}")
        print(f"  Total steps:        {total_steps:,}")
        print(f"  Per-GPU batch:      {args.batch_size}")
        print(f"  Grad accum:         {args.grad_accum}")
        print(f"  Effective batch:    {args.batch_size * args.grad_accum * world_size}")
        print(f"  Seq len:            {args.max_seq_len}")
        print(f"  Warmup:             {args.warmup_steps} steps")
        print(f"  LR:                 {args.lr}")
        if args.dataset == "mixed":
            source_mode = f"memmap ({args.data_bin})" if args.data_bin else f"streaming parquet ({args.data_dir})"
            print(f"  Data mode:          {source_mode}")
            print(
                f"  Curriculum:         preset={args.curriculum_preset} | "
                f"P1<{args.curriculum_phase1_ratio:.2f}, P2<{args.curriculum_phase2_ratio:.2f}, else P3"
            )
            print(f"  Oversample cap:     {args.max_oversample:.1f}x")
            repeat_policy = "allow restarts" if args.allow_source_restarts else "strict no-repeat"
            print(f"  Repeat policy:      {repeat_policy}")
            print(f"  Source warn @:      {args.source_epoch_warn:.2f}")
        else:
            print(f"  DataLoader:         workers={args.num_workers}, pin_memory={args.num_workers > 0}")


    
    raw_model = model.module if is_ddp else model

    if args.grad_checkpoint:
        raw_model.gradient_checkpointing_enable()
        if is_main_process(rank):
            print("[Memory] Gradient checkpointing: ON")
    else:
        raw_model.gradient_checkpointing_disable()
        if is_main_process(rank):
            print("[Memory] Gradient checkpointing: OFF")

    decay_params = []
    no_decay_params = []
    for name, param in raw_model.named_parameters():
        if param.requires_grad:
            if "bias" in name or "norm" in name or "log_prior" in name or "prior_weight" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    
    
    
    use_amp = args.fp16 and device.type == "cuda"
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    use_fp16 = use_amp and not use_bf16  
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)  

    if is_main_process(rank):
        if use_bf16:
            print(f"[Memory] Mixed Precision: BF16 (optimal for SNN)")
        elif use_fp16:
            print(f"[Memory] Mixed Precision: FP16 (fallback, BF16 not supported)")
        else:
            print(f"[Memory] Mixed Precision: OFF (FP32)")


    
    global_step = 0
    tokens_seen = 0
    best_loss = float("inf")
    train_log = []
    current_phase = 0
    resume_ckpt = None
    data_samples_seen = 0  

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt_path = save_dir / "latest.pt"
        if ckpt_path.exists():
            resume_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True, mmap=True)
            ckpt = resume_ckpt
            
            state = {k: v for k, v in ckpt["model"].items() if 'v_mem' not in k}
            raw_model.load_state_dict(state, strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            global_step = ckpt["global_step"]
            tokens_seen = ckpt.get("tokens_seen", global_step * tokens_per_step)
            best_loss = ckpt.get("best_loss", float("inf"))
            data_samples_seen = ckpt.get("data_samples_seen", 0)
            current_phase = ckpt.get("curriculum_phase", 0)

            
            
            
            
            
            
            
            if "spike_encoder_vmem" in ckpt:
                raw_model.spike_encoder.v_mem = ckpt["spike_encoder_vmem"].to(device)
                if is_main_process(rank):
                    print(f"[Resume] LIF membrane potential restored")
            else:
                if is_main_process(rank):
                    print(f"[Resume] WARNING: No LIF v_mem in checkpoint (old format).")
                    print(f"         Using 3-step warmup buffer to smooth transition...")

            if is_main_process(rank):
                print(f"[Resume] Loaded: step={global_step}, tokens_seen={tokens_seen / 1e9:.2f}B")
                print(f"[Resume] Data offset: {data_samples_seen} samples will be skipped")
        else:
            if is_main_process(rank):
                print(f"[Resume] No checkpoint found at {ckpt_path}, starting fresh")

    if args.dataset == "mixed":
        if is_main_process(rank):
            print(f"[DEBUG] Creating mixed dataset...", flush=True)
        dataset = build_mixed_dataset(
            args,
            rank,
            world_size,
            config.vocab_size,
            seed_offset=data_samples_seen,
        )
        restored_data_state = restore_checkpoint_data_state(
            dataset,
            resume_ckpt.get("data_state") if resume_ckpt else None,
            rank=rank,
            world_size=world_size,
            label="Resume",
        )
        phase_anchor = current_phase if restored_data_state else 0
        current_phase = maybe_switch_curriculum_phase(args, dataset, tokens_seen, phase_anchor, rank)
    elif is_main_process(rank):
        print(f"[DEBUG] Dataset created", flush=True)

    
    if is_main_process(rank):
        print(f"\nStarting training...\n")

    model.train()
    train_start = time.time()
    epoch_loss = 0.0
    epoch_steps = 0
    accum_loss = 0.0
    micro_step = 0  
    resume_warmup_remaining = 0  

    use_direct_dataset = args.dataset == "mixed"
    data_iter = None if use_direct_dataset else iter(dataloader)

    
    
    MAX_SKIP = 10000  
    skip_target = 0 if use_direct_dataset else data_samples_seen
    if skip_target > 0 and is_main_process(rank):
        print(f"[Resume] Skipping {skip_target} samples (of {data_samples_seen} total)...")
    skip_count = 0
    while skip_count < skip_target:
        try:
            next(data_iter)
            skip_count += 1
        except StopIteration:
            data_iter = iter(dataloader)
            continue
    if skip_target > 0 and is_main_process(rank):
        print(f"[Resume] Skipped {skip_count} samples, data stream aligned")

    
    
    if args.resume and global_step > 0:
        if 'ckpt' in dir() and "spike_encoder_vmem" not in ckpt:
            resume_warmup_remaining = 3
            if is_main_process(rank):
                print(f"[Resume] Warmup buffer: {resume_warmup_remaining} steps at 1/10 LR")

    while tokens_seen < args.total_tokens:
        
        if use_direct_dataset:
            current_phase = maybe_switch_curriculum_phase(args, dataset, tokens_seen, current_phase, rank)
            x, y = dataset.get_batch(args.batch_size, device)
        else:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

        
        lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr)
        
        if resume_warmup_remaining > 0:
            lr = lr * 0.1
            resume_warmup_remaining -= 1
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        data_samples_seen += args.batch_size  

        
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            main_loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1),
            )
            
            flat_logits = logits.reshape(-1, config.vocab_size)
            n_sample = min(128, flat_logits.size(0))
            z_idx = torch.randint(flat_logits.size(0), (n_sample,), device=logits.device)
            log_z = torch.logsumexp(flat_logits[z_idx], dim=-1)
            z_loss = 1e-4 * (log_z ** 2).mean()
            loss = (main_loss + z_loss) / args.grad_accum

        
        scaler.scale(loss).backward()
        accum_loss += main_loss.item()
        tokens_seen += x.numel() * world_size
        micro_step += 1

        
        if micro_step >= args.grad_accum:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            current_loss = accum_loss / args.grad_accum
            accum_loss = 0.0
            micro_step = 0  
            epoch_loss += current_loss
            epoch_steps += 1

            
            should_log = global_step % args.log_every == 0 and global_step > 0
            merged_recent_source_hist = {}
            merged_source_sampling = {}
            if should_log:
                local_recent_source_hist = (
                    dataset.consume_recent_source_histogram()
                    if use_direct_dataset and hasattr(dataset, "consume_recent_source_histogram")
                    else {}
                )
                local_source_sampling = (
                    dataset.get_source_sampling_stats()
                    if use_direct_dataset and hasattr(dataset, "get_source_sampling_stats")
                    else {}
                )
                if is_ddp:
                    gathered_stats = [None for _ in range(world_size)]
                    dist.all_gather_object(
                        gathered_stats,
                        {
                            "hist": local_recent_source_hist,
                            "sampling": local_source_sampling,
                        },
                    )
                else:
                    gathered_stats = [
                        {
                            "hist": local_recent_source_hist,
                            "sampling": local_source_sampling,
                        }
                    ]

                if is_main_process(rank):
                    merged_recent_source_hist = merge_source_histograms(
                        [item.get("hist", {}) for item in gathered_stats]
                    )
                    merged_source_sampling = merge_source_sampling_stats(
                        [item.get("sampling", {}) for item in gathered_stats]
                    )

            if should_log and is_main_process(rank):
                ppl = math.exp(min(current_loss, 20))
                elapsed = time.time() - train_start
                tokens_per_sec = tokens_seen / elapsed

                with torch.no_grad():
                    spikes, _ = raw_model.spike_encoder(x[:1, :32])
                    sparsity = 1.0 - spikes.mean().item()

                progress = tokens_seen / args.total_tokens * 100
                eta_seconds = (args.total_tokens - tokens_seen) / max(tokens_per_sec, 1)
                eta_hours = eta_seconds / 3600
                source_epoch_summary, source_epoch_warnings, source_epoch_snapshot = summarize_source_sampling_stats(
                    merged_source_sampling,
                    warn_threshold=args.source_epoch_warn,
                )

                gpu_info = f" | GPUs: {world_size}" if is_ddp else ""
                source_info = (
                    f" | Src: {format_source_histogram(merged_recent_source_hist)}"
                    if merged_recent_source_hist
                    else ""
                )
                source_epoch_info = f" | SrcEpoch: {source_epoch_summary}" if source_epoch_summary else ""
                print(f"Step {global_step:6d}/{total_steps} | "
                      f"Loss: {current_loss:.4f} | "
                      f"PPL: {ppl:8.1f} | "
                      f"LR: {lr:.2e} | "
                      f"Sparsity: {sparsity * 100:.1f}% | "
                      f"Tok/s: {tokens_per_sec:.0f} | "
                      f"Progress: {progress:.1f}% | "
                      f"ETA: {eta_hours:.1f}h{gpu_info}{source_info}{source_epoch_info}")
                if source_epoch_warnings:
                    print(f"  [SourceWarn] {', '.join(source_epoch_warnings)}")

                log_entry = {
                    "step": global_step,
                    "loss": current_loss,
                    "ppl": ppl,
                    "lr": lr,
                    "sparsity": sparsity,
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": tokens_per_sec,
                }

                if merged_recent_source_hist:
                    log_entry["source_histogram"] = merged_recent_source_hist
                if source_epoch_snapshot:
                    log_entry["source_sampling"] = source_epoch_snapshot
                    log_entry["source_sampling_warnings"] = source_epoch_warnings
                train_log.append(log_entry)

            
            should_rollback = False
            if current_loss > 15.0 and global_step > args.warmup_steps:
                should_rollback = True

            
            if is_ddp:
                rollback_tensor = torch.tensor([1.0 if should_rollback else 0.0], device=device)
                dist.all_reduce(rollback_tensor, op=dist.ReduceOp.MAX)
                should_rollback = rollback_tensor.item() > 0.5

            if should_rollback:
                if is_main_process(rank):
                    print(f"\n[ALERT] Loss spike detected: {current_loss:.2f} > 15.0!")
                    if is_ddp:
                        print("    At least one rank requested rollback; restoring all ranks to the latest checkpoint...")
                    print("    Rolling back to the latest checkpoint...")
                ckpt_path = save_dir / "latest.pt"
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True, mmap=True)
                    
                    state = {k: v for k, v in ckpt["model"].items() if 'v_mem' not in k}
                    raw_model.load_state_dict(state, strict=False)
                    optimizer.load_state_dict(ckpt["optimizer"])
                    scaler.load_state_dict(ckpt["scaler"])
                    global_step = ckpt["global_step"]
                    tokens_seen = ckpt.get("tokens_seen", global_step * tokens_per_step)
                    data_samples_seen = ckpt.get("data_samples_seen", 0)
                    current_phase = ckpt.get("curriculum_phase", current_phase)
                    
                    if "spike_encoder_vmem" in ckpt and ckpt["spike_encoder_vmem"] is not None:
                        raw_model.spike_encoder.v_mem = ckpt["spike_encoder_vmem"].to(device)
                    
                    min_lr = args.lr * 0.125 if not hasattr(args, '_initial_lr') else args._initial_lr * 0.125
                    args.lr = max(args.lr * 0.5, min_lr)
                    if is_main_process(rank):
                        print(f"    Rolled back to step {global_step}, LR -> {args.lr:.2e} (min: {min_lr:.2e})")
                    if is_ddp:
                        dist.barrier()  
                    if use_direct_dataset:
                        dataset = build_mixed_dataset(
                            args,
                            rank,
                            world_size,
                            config.vocab_size,
                            seed_offset=data_samples_seen,
                        )
                        restored_data_state = restore_checkpoint_data_state(
                            dataset,
                            ckpt.get("data_state"),
                            rank=rank,
                            world_size=world_size,
                            label="Rollback",
                        )
                        phase_anchor = current_phase if restored_data_state else 0
                        current_phase = maybe_switch_curriculum_phase(args, dataset, tokens_seen, phase_anchor, rank)
                    else:
                        data_iter, _ = realign_data_iterator(dataloader, data_samples_seen, rank, label="Rollback")
                    accum_loss = 0.0
                    accum_aux = 0.0
                    micro_step = 0
                    continue

            
            should_save = global_step % args.save_every == 0 and global_step > 0
            checkpoint_data_state = None
            if should_save:
                checkpoint_data_state = collect_checkpoint_data_state(
                    dataset,
                    use_direct_dataset=use_direct_dataset,
                    is_ddp=is_ddp,
                    rank=rank,
                    world_size=world_size,
                )
            if should_save and is_main_process(rank):
                _save_checkpoint(raw_model, optimizer, scaler, global_step, tokens_seen,
                                 best_loss, config, save_dir, train_log,
                                 data_samples_seen=data_samples_seen,
                                 curriculum_phase=current_phase,
                                 data_state=checkpoint_data_state,
                                 keep_n=args.keep_checkpoints)

            global_step += 1

    
    total_time = time.time() - train_start
    avg_loss = epoch_loss / max(epoch_steps, 1)
    final_ppl = math.exp(min(avg_loss, 20))
    final_data_state = collect_checkpoint_data_state(
        dataset,
        use_direct_dataset=use_direct_dataset,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size,
    )

    if is_main_process(rank):
        print(f"\n{'=' * 60}")
        print(f" Training Complete!")
        print(f"{'=' * 60}")
        print(f"Total steps:     {global_step:,}")
        print(f"Total tokens:    {tokens_seen / 1e9:.2f}B")
        print(f"Total time:      {total_time / 3600:.1f} hours")
        print(f"Avg tok/s:       {tokens_seen / total_time:.0f}")
        print(f"Final avg loss:  {avg_loss:.4f}")

        print(f"Final PPL:       {final_ppl:.2f}")

        
        _save_checkpoint(raw_model, optimizer, scaler, global_step, tokens_seen,
                         best_loss, config, save_dir, train_log,
                         data_samples_seen=data_samples_seen,
                         curriculum_phase=current_phase,
                         data_state=final_data_state)

        
        log_path = save_dir / "train_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(train_log, f, indent=2, ensure_ascii=False)
        print(f"Log saved to {log_path}")

    cleanup_distributed()


def _save_checkpoint(model, optimizer, scaler, step, tokens_seen,
                     best_loss, config, save_dir, train_log,
                     data_samples_seen=0, curriculum_phase=0, data_state=None, keep_n=3):
    
    
    
    
    
    
    
    
    raw_model = model.module if hasattr(model, 'module') else model
    ckpt = {
        
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": step,
        "tokens_seen": tokens_seen,
        "best_loss": best_loss,
        "config": config.__dict__,
        
        "spike_encoder_vmem": raw_model.spike_encoder.v_mem.cpu() if raw_model.spike_encoder.v_mem is not None else None,
        "data_samples_seen": data_samples_seen,
        "curriculum_phase": curriculum_phase,
        "data_state": data_state,
    }
    torch.save(ckpt, save_dir / "latest.pt")
    torch.save(ckpt, save_dir / f"step_{step}.pt")
    print(f"  [Checkpoint] Saved step {step} (tokens: {tokens_seen / 1e9:.2f}B, v_mem: saved)")

    
    if keep_n > 0:
        import glob as _glob
        saved = sorted(
            _glob.glob(str(save_dir / "step_*.pt")),
            key=lambda p: int(Path(p).stem.split('_')[1])
        )
        while len(saved) > keep_n:
            old = saved.pop(0)
            os.remove(old)
            print(f"  [Checkpoint] Deleted old: {Path(old).name}")





if __name__ == "__main__":
    args = parse_args()
    if 'RANK' not in os.environ or int(os.environ.get('RANK', 0)) == 0:
        print(f"\n{'=' * 60}")
        print(f" SymbolicLight V1 Training")
        print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")
        print(f"Config: {vars(args)}")
    train(args)
