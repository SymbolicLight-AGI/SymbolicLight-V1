#!/usr/bin/env python3
"""Public aggregate data pipeline for SymbolicLight 0.8B training.

The public release uses aggregate domain labels only. It does not encode
source-level dataset names, download URLs, or raw data manifests.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from train_tokenizer import SLTokenizer


DEFAULT_CURRICULUM_PRESET = "default"
DEFAULT_CURRICULUM_PHASE_1_RATIO = 0.20
DEFAULT_CURRICULUM_PHASE_2_RATIO = 0.65

PARQUET_TEXT_COLUMN_CANDIDATES = ("text", "translation")
QA_CONCAT_SENTINEL = "__qa_concat__"
SKIP_FILE_SENTINEL = "__skip_file__"


AGGREGATE_PHASES: dict[int, list[dict[str, Any]]] = {
    1: [
        {"name": "reference-web", "weight": 0.24, "text_key": "text"},
        {"name": "general-web", "weight": 0.20, "text_key": "text"},
        {"name": "academic-educational", "weight": 0.16, "text_key": "text"},
        {"name": "open-educational", "weight": 0.14, "text_key": "text"},
        {"name": "math-web", "weight": 0.12, "text_key": "text"},
        {"name": "code-text", "weight": 0.08, "text_key": "text"},
        {"name": "synthetic-narrative", "weight": 0.04, "text_key": "text"},
        {"name": "translation", "weight": 0.02, "text_key": "translation"},
    ],
    2: [
        {"name": "reference-web", "weight": 0.23, "text_key": "text"},
        {"name": "math-web", "weight": 0.18, "text_key": "text"},
        {"name": "code-text", "weight": 0.15, "text_key": "text"},
        {"name": "general-web", "weight": 0.15, "text_key": "text"},
        {"name": "academic-educational", "weight": 0.13, "text_key": "text"},
        {"name": "open-educational", "weight": 0.09, "text_key": "text"},
        {"name": "synthetic-narrative", "weight": 0.05, "text_key": "text"},
        {"name": "translation", "weight": 0.02, "text_key": "translation"},
    ],
    3: [
        {"name": "math-web", "weight": 0.24, "text_key": "text"},
        {"name": "code-text", "weight": 0.20, "text_key": "text"},
        {"name": "reference-web", "weight": 0.20, "text_key": "text"},
        {"name": "general-web", "weight": 0.12, "text_key": "text"},
        {"name": "academic-educational", "weight": 0.10, "text_key": "text"},
        {"name": "open-educational", "weight": 0.08, "text_key": "text"},
        {"name": "synthetic-narrative", "weight": 0.04, "text_key": "text"},
        {"name": "translation", "weight": 0.02, "text_key": "translation"},
    ],
}


def resolve_tokenizer_path(tokenizer_path: Optional[str]) -> str:
    here = Path(__file__).resolve().parent
    package_root = here.parent
    if not tokenizer_path:
        tokenizer_path = "sl_tokenizer.model"
    path = Path(tokenizer_path)
    candidates = [
        path,
        here / path,
        package_root / path,
        package_root / "tokenizer" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Tokenizer model not found: {tokenizer_path}. "
        f"Tried: {', '.join(str(candidate) for candidate in candidates)}"
    )


def normalize_source_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _recipe_aliases(recipe_name: str) -> set[str]:
    norm = normalize_source_id(recipe_name)
    aliases = {norm}
    aliases.add(normalize_source_id(recipe_name.replace("-", "_")))
    aliases.add(normalize_source_id(recipe_name.replace("_", "-")))
    return aliases


def resolve_recipe_source_weights(recipe: list[dict], available_source_ids: list[str]) -> tuple[dict[str, float], list[str]]:
    available = {normalize_source_id(source_id): source_id for source_id in available_source_ids}
    matched: dict[str, float] = {}
    unresolved: list[str] = []
    for source in recipe:
        resolved = [available[alias] for alias in _recipe_aliases(source["name"]) if alias in available]
        if not resolved:
            unresolved.append(str(source["name"]))
            continue
        share = float(source["weight"]) / len(resolved)
        for source_id in resolved:
            matched[source_id] = matched.get(source_id, 0.0) + share
    total = sum(matched.values())
    if total > 0:
        matched = {source_id: weight / total for source_id, weight in matched.items()}
    return matched, unresolved


def format_source_histogram(source_hist: dict) -> str:
    if not source_hist:
        return "n/a"
    total = max(float(sum(source_hist.values())), 1.0)
    parts = []
    for source_id, count in sorted(source_hist.items(), key=lambda item: item[1], reverse=True):
        parts.append(f"{source_id}:{float(count) / total * 100:.1f}%")
    return ", ".join(parts)


def get_curriculum_phase(
    tokens_seen: int,
    total_tokens: int,
    *,
    phase1_ratio: float = DEFAULT_CURRICULUM_PHASE_1_RATIO,
    phase2_ratio: float = DEFAULT_CURRICULUM_PHASE_2_RATIO,
    preset: str = DEFAULT_CURRICULUM_PRESET,
) -> tuple[int, list[dict[str, Any]]]:
    del preset
    progress = 0.0 if total_tokens <= 0 else float(tokens_seen) / float(total_tokens)
    if progress < phase1_ratio:
        phase = 1
    elif progress < phase2_ratio:
        phase = 2
    else:
        phase = 3
    return phase, [dict(item) for item in AGGREGATE_PHASES[phase]]


def extract_training_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = [extract_training_text(item) for item in value.values()]
        return "\n".join(part for part in parts if part)
    if isinstance(value, (list, tuple)):
        parts = [extract_training_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value).strip()


def resolve_parquet_text_column(parquet_module, path: Path) -> str:
    schema_names = set(parquet_module.ParquetFile(path).schema_arrow.names)
    for candidate in PARQUET_TEXT_COLUMN_CANDIDATES:
        if candidate in schema_names:
            return candidate
    if "question" in schema_names and "answers" in schema_names:
        return QA_CONCAT_SENTINEL
    return SKIP_FILE_SENTINEL


def parquet_columns_for_text_mode(text_column: str) -> list[str]:
    if text_column == QA_CONCAT_SENTINEL:
        return ["question", "answers"]
    return [text_column]


def extract_training_text_from_row(table, row_idx: int, text_column: str) -> str:
    if text_column == QA_CONCAT_SENTINEL:
        question = extract_training_text(table.column("question")[row_idx].as_py())
        answers = extract_training_text(table.column("answers")[row_idx].as_py())
        return "\n".join(part for part in (question, answers) if part)
    return extract_training_text(table.column(text_column)[row_idx].as_py())


class StreamingParquetDataset:
    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        *,
        seq_len: int = 512,
        rank: int = 0,
        world_size: int = 1,
        model_vocab_size: int = 57344,
        seed_offset: int = 0,
        max_oversample: float = 10.0,
        strict_no_repeat: bool = True,
    ):
        import pyarrow.parquet as pq

        self.pq = pq
        self.seq_len = int(seq_len)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_vocab_size = int(model_vocab_size)
        self.data_dir = str(data_dir)
        self.strict_no_repeat = bool(strict_no_repeat)
        self.max_oversample = float(max_oversample)
        self.rng = random.Random(20260331 + self.rank + int(seed_offset))
        self.tokenizer = SLTokenizer(resolve_tokenizer_path(tokenizer_path))
        self._buffer: list[int] = []
        self._recent_source_hist: dict[str, int] = {}
        self._source_tokenized_tokens: dict[str, int] = {}
        self._source_sampled_train_tokens: dict[str, int] = {}
        self._source_files_completed: dict[str, int] = {}
        self._phase_weights: dict[str, float] = {}
        self._phase_unresolved: list[str] = []
        self._file_offsets: dict[str, int] = {}
        self._text_column_cache: dict[Path, str] = {}

        root = Path(data_dir)
        self._source_index: dict[str, list[Path]] = {}
        for path in sorted(root.rglob("*.parquet")):
            source_id = normalize_source_id(path.parent.name)
            self._source_index.setdefault(source_id, []).append(path)
        if not self._source_index:
            raise FileNotFoundError(f"No .parquet files found under {data_dir}")
        for source_id in self._source_index:
            self.rng.shuffle(self._source_index[source_id])
            self._file_offsets[source_id] = 0
        self._source_total_files = {source_id: len(files) for source_id, files in self._source_index.items()}
        self._source_natural_weights = self._compute_natural_weights()

    def _compute_natural_weights(self) -> dict[str, float]:
        sizes = {
            source_id: sum(path.stat().st_size for path in files)
            for source_id, files in self._source_index.items()
        }
        total = sum(sizes.values()) or 1
        return {source_id: size / total for source_id, size in sizes.items()}

    def set_phase(self, recipe: list[dict]) -> None:
        weights, unresolved = resolve_recipe_source_weights(recipe, list(self._source_index))
        if not weights:
            weights = {source_id: 1.0 / len(self._source_index) for source_id in self._source_index}
        if self.max_oversample > 0:
            for source_id, weight in list(weights.items()):
                cap = self._source_natural_weights.get(source_id, 0.0) * self.max_oversample
                if cap > 0 and weight > cap:
                    weights[source_id] = cap
            total = sum(weights.values())
            if total > 0:
                weights = {source_id: weight / total for source_id, weight in weights.items()}
        self._phase_weights = weights
        self._phase_unresolved = unresolved

    def get_phase_summary(self) -> dict:
        return {
            "active_sources": dict(self._phase_weights),
            "unresolved_sources": list(self._phase_unresolved),
            "available_sources": sorted(self._source_index),
            "strict_no_repeat": self.strict_no_repeat,
        }

    def _choose_source(self) -> str:
        active = [(source_id, weight) for source_id, weight in self._phase_weights.items() if source_id in self._source_index]
        if not active:
            active = [(source_id, 1.0) for source_id in self._source_index]
        source_ids, weights = zip(*active)
        return self.rng.choices(list(source_ids), weights=list(weights), k=1)[0]

    def _next_file(self, source_id: str) -> Path:
        files = self._source_index[source_id]
        offset = self._file_offsets.get(source_id, 0)
        if offset >= len(files):
            if self.strict_no_repeat:
                raise RuntimeError(f"Source exhausted under strict no-repeat policy: {source_id}")
            self.rng.shuffle(files)
            offset = 0
        self._file_offsets[source_id] = offset + 1
        self._recent_source_hist[source_id] = self._recent_source_hist.get(source_id, 0) + 1
        return files[offset]

    def _fill_buffer(self) -> None:
        max_attempts = max(16, len(self._source_index) * 4)
        for _ in range(max_attempts):
            source_id = self._choose_source()
            path = self._next_file(source_id)
            text_column = self._text_column_cache.get(path)
            if text_column is None:
                text_column = resolve_parquet_text_column(self.pq, path)
                self._text_column_cache[path] = text_column
            if text_column == SKIP_FILE_SENTINEL:
                continue
            table = self.pq.read_table(path, columns=parquet_columns_for_text_mode(text_column))
            for row_idx in range(len(table)):
                text = extract_training_text_from_row(table, row_idx, text_column)
                if text and len(text) > 10:
                    ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
                    self._buffer.extend(ids)
                    self._source_tokenized_tokens[source_id] = self._source_tokenized_tokens.get(source_id, 0) + len(ids)
                    if len(self._buffer) >= self.seq_len + 1:
                        return
            self._source_files_completed[source_id] = self._source_files_completed.get(source_id, 0) + 1
        raise RuntimeError(f"Unable to fill token buffer from parquet files under {self.data_dir}")

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []
        for _ in range(int(batch_size)):
            while len(self._buffer) < self.seq_len + 1:
                self._fill_buffer()
            tokens = self._buffer[: self.seq_len + 1]
            self._buffer = self._buffer[self.seq_len :]
            rows.append(tokens)
        x = torch.tensor(rows, dtype=torch.long, device=device).clamp(0, self.model_vocab_size - 1)
        return x[:, :-1], x[:, 1:]

    def consume_recent_source_histogram(self) -> dict:
        hist = dict(self._recent_source_hist)
        self._recent_source_hist = {}
        return hist

    def get_source_sampling_stats(self) -> dict:
        sources = {}
        for source_id in sorted(self._source_index):
            sources[source_id] = {
                "mode": "streaming_observed",
                "sampled_train_tokens": int(self._source_sampled_train_tokens.get(source_id, 0)),
                "unique_token_budget": int(self._source_tokenized_tokens.get(source_id, 0)),
                "budget_is_exact": False,
                "completed_files": int(self._source_files_completed.get(source_id, 0)),
                "total_files": int(self._source_total_files.get(source_id, 0)),
                "active": source_id in self._phase_weights,
            }
        return {"mode": "streaming_observed", "sources": sources}

    def state_dict(self) -> dict:
        return {
            "buffer": list(self._buffer),
            "phase_weights": dict(self._phase_weights),
            "phase_unresolved": list(self._phase_unresolved),
            "file_offsets": dict(self._file_offsets),
            "rng_state": self.rng.getstate(),
            "recent_source_hist": dict(self._recent_source_hist),
            "source_tokenized_tokens": dict(self._source_tokenized_tokens),
            "source_files_completed": dict(self._source_files_completed),
        }

    def load_state_dict(self, state: Optional[dict]) -> None:
        if not state:
            return
        self._buffer = list(state.get("buffer", []))
        self._phase_weights = dict(state.get("phase_weights", {}))
        self._phase_unresolved = list(state.get("phase_unresolved", []))
        self._file_offsets.update({str(k): int(v) for k, v in state.get("file_offsets", {}).items()})
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])
        self._recent_source_hist = dict(state.get("recent_source_hist", {}))
        self._source_tokenized_tokens = dict(state.get("source_tokenized_tokens", {}))
        self._source_files_completed = dict(state.get("source_files_completed", {}))


class MemmapDataset:
    def __init__(
        self,
        data_bin_dir: str,
        *,
        seq_len: int = 512,
        rank: int = 0,
        world_size: int = 1,
        model_vocab_size: int = 57344,
        seed_offset: int = 0,
        strict_no_repeat: bool = True,
    ):
        del strict_no_repeat
        self.data_bin_dir = Path(data_bin_dir)
        self.seq_len = int(seq_len)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_vocab_size = int(model_vocab_size)
        self.rng = random.Random(20260331 + self.rank + int(seed_offset))
        train_bin = self.data_bin_dir / "train.bin"
        if not train_bin.exists():
            raise FileNotFoundError(f"Missing pretokenized file: {train_bin}")
        dtype = np.uint16
        meta_path = self.data_bin_dir / "train.meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if str(meta.get("dtype", "")).lower() in {"uint32", "np.uint32"}:
                dtype = np.uint32
        self.tokens = np.memmap(train_bin, dtype=dtype, mode="r")
        if len(self.tokens) <= self.seq_len + 1:
            raise ValueError(f"train.bin is too small for seq_len={self.seq_len}")
        self._phase_recipe: list[dict] = []

    def set_phase(self, recipe: list[dict]) -> None:
        self._phase_recipe = [dict(item) for item in recipe]

    def get_phase_summary(self) -> dict:
        return {"active_sources": {item["name"]: item["weight"] for item in self._phase_recipe}, "unresolved_sources": []}

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.tokens) - self.seq_len - 1
        starts = [self.rng.randint(0, max_start) for _ in range(int(batch_size))]
        rows = [np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64) for start in starts]
        x = torch.tensor(np.stack(rows), dtype=torch.long, device=device).clamp(0, self.model_vocab_size - 1)
        return x[:, :-1], x[:, 1:]

    def consume_recent_source_histogram(self) -> dict:
        return {}

    def get_source_sampling_stats(self) -> dict:
        return {"mode": "memmap_public", "sources": {}}

    def state_dict(self) -> dict:
        return {"rng_state": self.rng.getstate(), "phase_recipe": list(self._phase_recipe)}

    def load_state_dict(self, state: Optional[dict]) -> None:
        if not state:
            return
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])
        self._phase_recipe = list(state.get("phase_recipe", []))
