from collections import defaultdict
import token
import torch
from dataclasses import dataclass
from collections import defaultdict


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
    data: torch.Tensor, batch_size: int, time_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    high_idx = len(data) - time_size
    ind = torch.randint(low=0, high=high_idx, size=(batch_size,))
    grid = ind[:, None] + torch.arange(time_size)

    xb = data[grid]
    yb = data[grid + 1]

    return (xb, yb)


class Tokenizer:
    TARGET_VOCAB_SIZE = 1000

    def __init__(self):
        self.next_id = 256
        self.bpe_mapping: dict[int, tuple[int, int]] = {}
        self.vocab_size: int = -1

    def encode_bpe(self) -> torch.Tensor:
        pair_vocab: dict[int, tuple[int, int]] = {}
        with open("input.txt", "r", encoding="utf-8") as f:
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
        self.vocab_size = self.next_id
        print(f"vocab size: {self.vocab_size}")
        return torch.tensor(token_list)

    def get_vocab_size(self) -> int:
        print(f"vocab size: {self.vocab_size}")
        return self.vocab_size

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

        print(f"encoded shape: {encoded.shape}")

        return DataSplit(
            train=encoded[:train_n],
            val=encoded[train_n : train_n + val_n],
            test=encoded[train_n + val_n :],
        )
