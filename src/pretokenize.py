#!/usr/bin/env python3
"""
Pretokenize parquet data into a source-aware memmap for 0.8B training.

Outputs:
  - train.bin
  - train.meta.json (includes source_stats and source_spans)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline import (
    SKIP_FILE_SENTINEL,
    extract_training_text_from_row,
    parquet_columns_for_text_mode,
    resolve_parquet_text_column,
    resolve_tokenizer_path,
)
from train_tokenizer import SLTokenizer


_worker_tokenizer = None


def _resolve_source_name(data_root: Path, parquet_path: Path) -> str:
    try:
        relative = parquet_path.relative_to(data_root)
        if relative.parts:
            return relative.parts[0]
    except ValueError:
        pass
    return parquet_path.parent.name


def _init_worker(tokenizer_path: str) -> None:
    global _worker_tokenizer
    _worker_tokenizer = SLTokenizer(resolve_tokenizer_path(tokenizer_path))


def _tokenize_table_rows(table, text_column: str, *, min_text_len: int) -> list[list[int]]:
    docs = []
    for row_idx in range(len(table)):
        text = extract_training_text_from_row(table, row_idx, text_column)
        if text and len(text) >= min_text_len:
            docs.append(_worker_tokenizer.encode(text, add_bos=True, add_eos=True))
    return docs


def _count_file_tokens(args) -> tuple[str, str, int, int]:
    file_path_str, source_name, min_text_len = args
    try:
        import pyarrow.parquet as pq

        file_path = Path(file_path_str)
        parquet_file = pq.ParquetFile(file_path)
        text_column = resolve_parquet_text_column(pq, file_path)
        if text_column == SKIP_FILE_SENTINEL:
            return file_path_str, source_name, 0, 0

        total_tokens = 0
        columns = parquet_columns_for_text_mode(text_column)
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=columns)
            docs = _tokenize_table_rows(table, text_column, min_text_len=min_text_len)
            total_tokens += sum(len(ids) for ids in docs)

        return file_path_str, source_name, total_tokens, parquet_file.num_row_groups
    except Exception as exc:
        print(f"[Pass 1 WARNING] skipping {file_path_str}: {exc}", flush=True)
        return file_path_str, source_name, 0, 0


def _tokenize_row_group(args) -> tuple[str, str, int, list[list[int]], int]:
    file_path_str, source_name, row_group_idx, min_text_len = args
    try:
        import pyarrow.parquet as pq

        file_path = Path(file_path_str)
        parquet_file = pq.ParquetFile(file_path)
        text_column = resolve_parquet_text_column(pq, file_path)
        if text_column == SKIP_FILE_SENTINEL:
            return file_path_str, source_name, int(row_group_idx), [], 0

        table = parquet_file.read_row_group(
            int(row_group_idx),
            columns=parquet_columns_for_text_mode(text_column),
        )
        docs = _tokenize_table_rows(table, text_column, min_text_len=min_text_len)
        total_tokens = sum(len(ids) for ids in docs)
        return file_path_str, source_name, int(row_group_idx), docs, total_tokens
    except Exception as exc:
        print(
            f"[Pass 2 WARNING] skipping row_group {row_group_idx} of {file_path_str}: {exc}",
            flush=True,
        )
        return file_path_str, source_name, int(row_group_idx), [], 0


def _write_meta(
    meta_path: Path,
    *,
    total_tokens: int,
    vocab_size: int,
    tokenizer_model: str,
    source_stats: dict,
    source_spans: dict,
) -> None:
    meta = {
        "total_tokens": int(total_tokens),
        "dtype": "uint16",
        "vocab_size": int(vocab_size),
        "active_vocab_size": int(vocab_size),
        "tokenizer_model": str(tokenizer_model),
        "source_stats": source_stats,
        "source_spans": source_spans,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretokenize parquet data into memmap for 0.8B training")
    parser.add_argument("--data_dir", type=str, required=True, help="Parquet data root")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for train.bin/train.meta.json")
    parser.add_argument("--tokenizer", type=str, default="../tokenizer/sl_tokenizer.model", help="Tokenizer model path")
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 8), help="Worker processes")
    parser.add_argument("--min_text_len", type=int, default=10, help="Minimum text length before tokenization")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_model = resolve_tokenizer_path(args.tokenizer)
    tokenizer = SLTokenizer(tokenizer_model)
    vocab_size = int(tokenizer.vocab_size)
    if vocab_size > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Tokenizer vocab_size={vocab_size} exceeds uint16 range; upgrade the binary format first."
        )

    all_files = sorted(data_dir.rglob("*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No .parquet files found under {data_dir}")

    print(f"[Config] data_dir={data_dir}")
    print(f"[Config] output_dir={output_dir}")
    print(f"[Config] tokenizer={tokenizer_model} | vocab_size={vocab_size}")
    print(f"[Data] found {len(all_files)} parquet files")

    count_work_items = [
        (str(file_path), _resolve_source_name(data_dir, file_path), int(args.min_text_len))
        for file_path in all_files
    ]

    total_tokens = 0
    row_group_work_items = []
    source_token_counts = {}
    t0 = time.time()

    with Pool(processes=max(1, int(args.workers)), initializer=_init_worker, initargs=(tokenizer_model,)) as pool:
        for idx, (file_path_str, source_name, file_tokens, num_row_groups) in enumerate(
            pool.imap_unordered(_count_file_tokens, count_work_items, chunksize=1),
            start=1,
        ):
            total_tokens += int(file_tokens)
            source_token_counts[source_name] = source_token_counts.get(source_name, 0) + int(file_tokens)
            row_group_work_items.extend(
                (file_path_str, source_name, row_group_idx, int(args.min_text_len))
                for row_group_idx in range(int(num_row_groups))
            )
            if idx % 50 == 0 or idx == len(count_work_items):
                elapsed = time.time() - t0
                print(
                    f"[Pass 1] files={idx}/{len(count_work_items)} | "
                    f"tokens={total_tokens / 1e9:.2f}B | elapsed={elapsed / 60:.1f} min"
                )

    if total_tokens <= 0:
        raise RuntimeError("No valid training tokens produced from the parquet directory.")

    bin_path = output_dir / "train.bin"
    meta_path = output_dir / "train.meta.json"
    print(f"[Pass 2] writing memmap to {bin_path}")
    mmap = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(int(total_tokens),))

    offset = 0
    source_stats = {}
    source_spans = {}
    t1 = time.time()

    with Pool(processes=max(1, int(args.workers)), initializer=_init_worker, initargs=(tokenizer_model,)) as pool:
        for idx, (_file_path_str, source_name, _row_group_idx, docs, rg_tokens) in enumerate(
            pool.imap_unordered(_tokenize_row_group, row_group_work_items, chunksize=1),
            start=1,
        ):
            if not docs or int(rg_tokens) <= 0:
                continue

            span_start = offset
            for ids in docs:
                arr = np.asarray(ids, dtype=np.uint16)
                mmap[offset : offset + len(arr)] = arr
                offset += len(arr)

            source_spans.setdefault(source_name, []).append([int(span_start), int(offset)])
            source_stats[source_name] = source_stats.get(source_name, 0) + int(rg_tokens)

            if idx % 200 == 0 or idx == len(row_group_work_items):
                elapsed = time.time() - t1
                print(
                    f"[Pass 2] row_groups={idx}/{len(row_group_work_items)} | "
                    f"written={offset / 1e9:.2f}B tokens | elapsed={elapsed / 60:.1f} min"
                )

    if offset != total_tokens:
        print(f"[Pass 2] WARNING: planned {total_tokens} tokens but wrote {offset}; truncating metadata to actual size")
        total_tokens = int(offset)

    mmap.flush()
    del mmap

    _write_meta(
        meta_path,
        total_tokens=int(total_tokens),
        vocab_size=vocab_size,
        tokenizer_model=tokenizer_model,
        source_stats=source_stats,
        source_spans=source_spans,
    )

    elapsed = time.time() - t0
    print(f"[Done] train.bin={bin_path}")
    print(f"[Done] train.meta.json={meta_path}")
    print(f"[Done] total_tokens={int(total_tokens):,} | elapsed={elapsed / 3600:.2f}h")
    print("[Done] source distribution:")
    for source_name, count in sorted(source_stats.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {source_name}: {count / max(int(total_tokens), 1) * 100:.2f}% ({count:,} tokens)")


if __name__ == "__main__":
    main()
