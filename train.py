import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import override
from helpers import (
    DataSplit,
    EvalSplit,
    Tokenizer,
    extract_input,
    extract_vocab,
    generate_batch,
)

# --- Hyperparameters ---
vocab_size = -1
time_size = 8
batch_size = 32
n_embd = 32
training_steps = 10000
lr = 1e-3
eval_iters = training_steps // 10
eval_interval = training_steps // 10

tokenizer: Tokenizer = Tokenizer()


def extract_data() -> DataSplit:
    global vocab_size

    input = extract_input()
    vocab = extract_vocab()
    vocab_size = len(vocab)
    tokenizer.generate_tokenizer(vocab)

    data = tokenizer.get_data_split(input, val_percentage=0.1, test_percentage=0.1)
    return data


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table: nn.Embedding = nn.Embedding(vocab_size, n_embd)
        self.lm_head: nn.Linear = nn.Linear(n_embd, vocab_size)

    @override
    def forward(
        self, batch: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        token_embd = self.embedding_table(batch)
        logits = self.lm_head(token_embd)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            targets = targets.view(B * T)
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets)

        return (logits, loss)

    def generate(self, ctx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        for _ in range(max_tokens):
            logits, _ = self(ctx)
            logits = logits[:, -1, :]  # Isolate the last time dimension -> (B, C)
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            ctx = torch.cat((ctx, next_token), dim=1)

        return ctx

    @torch.no_grad
    def evaluate_loss(self, data_split: DataSplit) -> EvalSplit:
        _ = self.eval()
        eval_split = EvalSplit()

        # Train eval
        losses = torch.zeros(eval_iters)
        for step in range(eval_iters):
            xb, yb = generate_batch(data_split.train, batch_size, time_size)
            _, loss = self(xb, yb)
            losses[step] = loss
        eval_split.train_loss = losses.mean(dim=0).item()

        # Val eval
        losses = torch.zeros(eval_iters)
        for step in range(eval_iters):
            xb, yb = generate_batch(data_split.val, batch_size, time_size)
            _, loss = self(xb, yb)
            losses[step] = loss
        eval_split.val_loss = losses.mean(dim=0).item()

        _ = self.train()

        return eval_split


def train() -> BigramLanguageModel:
    data_split = extract_data()
    bigram_model = BigramLanguageModel()

    optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=lr)
    for step in range(training_steps):
        xb, yb = generate_batch(data_split.train, batch_size, time_size)
        _, loss = bigram_model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            eval_split = bigram_model.evaluate_loss(data_split)
            print(
                f"Iterations: {step}\tTraining loss = {eval_split.train_loss}\tValidation loss = {eval_split.val_loss}\n"
            )

    return bigram_model


if __name__ == "__main__":
    bigram_model = train()

    init_ctx = torch.zeros((1, 1), dtype=torch.long)
    print(tokenizer.decode_stream(bigram_model.generate(init_ctx, max_tokens=500)))
