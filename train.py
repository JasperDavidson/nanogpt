import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import cast
from typing_extensions import override
from helpers import DataSplit, Tokenizer, extract_input, extract_vocab, generate_batch

vocab_size = 0


def extract_data() -> DataSplit:
    global vocab_size

    input = extract_input()
    vocab = extract_vocab()
    vocab_size = len(vocab)
    t = Tokenizer()
    t.generate_tokenizer(vocab)

    data = t.get_data_split(input, val_percentage=0.1, test_percentage=0.1)
    return data


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding_table: nn.Embedding = nn.Embedding(vocab_size, vocab_size)

    @override
    def forward(
        self, batch: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = cast(torch.Tensor, self.embedding_table(batch))

        B, T, C = logits.shape
        logits = logits.view(B * T, C)

        loss = None
        if targets:
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return (logits, loss)

    def generate(self, ctx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        for _ in range(max_tokens):
            logits, _ = cast(tuple[torch.Tensor, torch.Tensor], self(ctx))
            logits = logits[:, -1, :]  # Isolate the last time dimension -> (B, C)
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            ctx = torch.cat((ctx, next_token), dim=1)

        return ctx


def train():
    data_split = extract_data()
    bigram_model = BigramLanguageModel(vocab_size)

    xb, yb = generate_batch(data_split.train, 4, 8)
    logits = cast(tuple[torch.Tensor, torch.Tensor], bigram_model(xb, yb))
