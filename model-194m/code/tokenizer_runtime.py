"""SentencePiece runtime wrapper for the released 194M tokenizer."""

from pathlib import Path
from typing import Iterable

import sentencepiece as spm


class SLTokenizer:
    def __init__(self, model_path: str | Path):
        path = Path(model_path)
        try:
            self.sp = spm.SentencePieceProcessor(model_file=str(path))
        except OSError:
            self.sp = spm.SentencePieceProcessor(model_proto=path.read_bytes())

    @property
    def vocab_size(self) -> int:
        return int(self.sp.vocab_size())

    @property
    def bos_id(self) -> int | None:
        value = int(self.sp.bos_id())
        return value if value >= 0 else None

    @property
    def eos_id(self) -> int | None:
        value = int(self.sp.eos_id())
        return value if value >= 0 else None

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = list(self.sp.encode(str(text), out_type=int))
        if add_bos and self.bos_id is not None:
            ids.insert(0, self.bos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.sp.decode([int(token_id) for token_id in ids])
