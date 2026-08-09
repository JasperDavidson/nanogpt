import torch
from typing import cast
from dataclasses import dataclass


def extract_input() -> str:
    with open("input.txt", "r", encoding="utf-8") as f:
        return f.read()


def extract_vocab() -> list[str]:
    with open("input.txt", "r", encoding="utf-8") as f:
        return sorted(list(set(f.read())))


@dataclass(slots=True)
class DataSplit:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor

torch.manual_seed(1337) # temp for watching

def generate_batch(
    data: torch.Tensor, batch_size: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    high_idx = len(data) - block_size
    ind = torch.randint(low=0, high=high_idx, size=(batch_size,))
    grid = ind[:, None] + torch.arange(block_size)

    xb = data[grid]
    yb = data[grid + 1]

    return (xb, yb)


class Tokenizer:
    def __init__(self):
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def generate_tokenizer(self, vocab: list[str]):
        self.stoi = {v: i for i, v in enumerate(vocab)}
        self.itos = {i: v for i, v in enumerate(vocab)}

    # Assumes a two dimensional (batch, text) input tensor
    def decode_stream(self, input: torch.Tensor) -> str:
        char_indices = cast(list[int], input.view(-1).tolist())  # pyright: ignore[reportUnknownMemberType]

        return "".join(self.itos[idx] for idx in char_indices)

    def get_data_split(
        self, input: str, val_percentage: float, test_percentage: float
    ) -> DataSplit:
        encoded = torch.tensor([self.stoi[c] for c in input], dtype=torch.long)
        n = len(encoded)
        test_n = int(n * test_percentage)
        val_n = int(n * val_percentage)
        train_n = n - test_n - val_n

        return DataSplit(
            train=encoded[:train_n],
            val=encoded[train_n : train_n + val_n],
            test=encoded[train_n + val_n :],
        )
