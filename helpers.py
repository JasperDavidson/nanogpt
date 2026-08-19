from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json

import torch


@dataclass(slots=True)
class DataSplit:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


@dataclass(slots=True)
class EvalSplit:
    train_loss: float = 0
    val_loss: float = 0


torch.manual_seed(1337)  # temp for watching


def generate_batch(
    data: torch.Tensor, d_batch: int, d_time: int
) -> tuple[torch.Tensor, torch.Tensor]:
    high_idx = len(data) - d_time
    ind = torch.randint(low=0, high=high_idx, size=(d_batch,))
    grid = ind[:, None] + torch.arange(d_time)

    xb = data[grid]
    yb = data[grid + 1]

    return (xb, yb)


class Tokenizer:
    TARGET_VOCAB_SIZE = 1000
    SOURCE_PATH = "input.txt"
    CACHE_PATH = Path("tokenizer_cache.json")

    def __init__(self):
        self.next_id = 256
        self.bpe_mapping: dict[int, tuple[int, int]] = {}
        self.d_vocab: int = -1

    def _load_cache(self) -> torch.Tensor | None:
        if not self.CACHE_PATH.exists():
            return None
        try:
            payload = json.loads(self.CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            payload.get("source") != self.SOURCE_PATH
            or payload.get("max_vocab_size") != self.TARGET_VOCAB_SIZE
        ):
            return None
        mapping = payload.get("bpe_mapping")
        tokens = payload.get("tokens")
        next_id = payload.get("next_id")
        d_vocab = payload.get("d_vocab")
        if not isinstance(mapping, dict) or not isinstance(tokens, list):
            return None
        if not isinstance(next_id, int) or not isinstance(d_vocab, int):
            return None
        self.bpe_mapping = {int(k): (int(v[0]), int(v[1])) for k, v in mapping.items()}
        self.next_id = next_id
        self.d_vocab = d_vocab
        return torch.tensor(tokens)

    def _save_cache(self, tokens: list[int]) -> None:
        payload = {
            "source": self.SOURCE_PATH,
            "max_vocab_size": self.TARGET_VOCAB_SIZE,
            "tokens": tokens,
            "bpe_mapping": {str(k): list(v) for k, v in self.bpe_mapping.items()},
            "next_id": self.next_id,
            "d_vocab": self.d_vocab,
        }
        self.CACHE_PATH.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def encode_bpe(self) -> torch.Tensor:
        cached = self._load_cache()
        if cached is not None:
            return cached

        pair_vocab: dict[int, tuple[int, int]] = {}
        with open(self.SOURCE_PATH, "r", encoding="utf-8") as f:
            token_list = list(f.read().encode("utf-8"))

        while self.next_id < self.TARGET_VOCAB_SIZE:
            pair_counts: dict[tuple[int, int], int] = defaultdict(int)
            for i in range(1, len(token_list)):
                prev_byte = token_list[i - 1]
                cur_byte = token_list[i]
                cur_pair = (prev_byte, cur_byte)

                pair_counts[cur_pair] += 1

            next_pair, count = max(
                pair_counts.items(), key=lambda item: item[1], default=(None, 0)
            )
            if count == 0:
                break
            assert next_pair is not None
            pair_vocab[self.next_id] = next_pair

            bpe_list: list[int] = []
            i = 0
            while i < len(token_list):
                if (
                    i + 1 < len(token_list)
                    and (token_list[i], token_list[i + 1]) == next_pair
                ):
                    bpe_list.append(self.next_id)
                    i += 2
                else:
                    bpe_list.append(token_list[i])
                    i += 1
            token_list = bpe_list

            self.next_id += 1

        self.bpe_mapping = pair_vocab
        self.d_vocab = self.next_id
        self._save_cache(token_list)
        return torch.tensor(token_list)

    def get_vocab_size(self) -> int:
        return self.d_vocab

    def expand_encoding(self, encoding: int) -> list[int]:
        if encoding <= 255:
            return [encoding]

        pair = self.bpe_mapping[encoding]
        if pair[0] <= 255 and pair[1] <= 255:
            return list(pair)

        pair_expanded = []
        pair_expanded.extend(self.expand_encoding(pair[0]))
        pair_expanded.extend(self.expand_encoding(pair[1]))

        return pair_expanded

    # Assumes a two dimensional (batch, text) input tensor
    def decode_stream(self, input: torch.Tensor) -> str:
        char_indices: list[int] = input.view(-1).tolist()
        res_indices = [d_c for c in char_indices for d_c in self.expand_encoding(c)]

        return bytes(res_indices).decode("utf-8")

    def get_data_split(
        self, val_percentage: float, test_percentage: float
    ) -> DataSplit:
        encoded = self.encode_bpe()
        n = len(encoded)
        test_n = int(n * test_percentage)
        val_n = int(n * val_percentage)
        train_n = n - test_n - val_n

        return DataSplit(
            train=encoded[:train_n],
            val=encoded[train_n : train_n + val_n],
            test=encoded[train_n + val_n :],
        )
