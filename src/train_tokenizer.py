#!/usr/bin/env python3
"""Public tokenizer wrapper for SymbolicLight.

This release file intentionally contains no corpus download or tokenizer-training
logic. It only loads the released SentencePiece model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import sentencepiece as spm


def _resolve_model_path(model_path: str | Path | None = None) -> str:
    here = Path(__file__).resolve().parent
    package_root = here.parent
    candidates = []
    if model_path:
        path = Path(model_path)
        candidates.extend([path, here / path, package_root / path])
    candidates.extend(
        [
            package_root / "tokenizer" / "sl_tokenizer.model",
            here / "sl_tokenizer.model",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"SentencePiece model not found. Tried: {tried}")


class SLTokenizer:
    """Small compatibility wrapper around SentencePieceProcessor."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = _resolve_model_path(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.vocab_size = int(self.sp.vocab_size())
        self.bos_id = int(self.sp.bos_id()) if self.sp.bos_id() >= 0 else None
        self.eos_id = int(self.sp.eos_id()) if self.sp.eos_id() >= 0 else None

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.sp.encode(str(text), out_type=int))
        if add_bos and self.bos_id is not None:
            ids.insert(0, self.bos_id)
        if add_eos and self.eos_id is not None:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.sp.decode([int(idx) for idx in ids])

    def __len__(self) -> int:
        return self.vocab_size
