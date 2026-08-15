from pyexpat import model
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import override
from helpers import (
    DataSplit,
    EvalSplit,
    Tokenizer,
    generate_batch,
)

# --- Hyperparameters ---
d_vocab = -1
d_time = 64
d_batch = 32
d_model = 128
training_steps = 100000
lr = 1e-3
eval_iters = training_steps // 10
eval_interval = training_steps // 10
n_heads = 4
dropout = 0.2

tokenizer: Tokenizer = Tokenizer()


def extract_data() -> DataSplit:
    global d_vocab

    data = tokenizer.get_data_split(val_percentage=0.1, test_percentage=0.1)
    d_vocab = tokenizer.get_vocab_size()

    return data


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        self.d_head = int(d_model / n_heads)
        self.n_heads = n_heads

        self.layer_norm = nn.LayerNorm((d_model))
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.output = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        d_batch = input.shape[0]
        d_time = input.shape[1]
        d_model = input.shape[2]

        ln_input = self.layer_norm(input)
        q = (
            self.query.forward(ln_input)
            .view(d_batch, d_time, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        k = (
            self.key.forward(ln_input)
            .view(d_batch, d_time, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        v = (
            self.value.forward(ln_input)
            .view(d_batch, d_time, self.n_heads, self.d_head)
            .transpose(1, 2)
        )

        affinity = (
            q @ k.transpose(-2, -1)
        )  # Note only transpose along (time, feature) dimension; attention is not cross-batch

        tril = torch.tril(torch.ones(self.n_heads, d_time, d_time))
        affinity = affinity.masked_fill(tril == 0, float("-inf"))
        affinity *= (
            1 / (self.d_head**0.5)
        )  # Reduce the variance after d_head ~mean=0, variance=1 elements accumulate through dot
        affinity = F.softmax(
            affinity, dim=-1
        )  # Only softmax across the feature dimension
        affinity = self.attn_dropout(affinity)

        a_out = (affinity @ v).transpose(1, 2).contiguous().view(d_batch, d_time, d_model)
        return self.resid_dropout(self.output.forward(a_out))


class NormHidden(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm((d_model))
        self.relu = nn.ReLU()
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ln_input = self.layer_norm(input)
        hidden_out = self.relu(self.linear(ln_input))

        return self.resid_dropout(hidden_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_time: int, d_model: int, n_heads: int):
        super().__init__()

        self.attn = SelfAttention(d_model, n_heads)
        self.ffn = NormHidden(d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attention = self.attn(input) + input
        hidden = self.ffn(attention) + attention

        return hidden


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embedding_table: nn.Embedding = nn.Embedding(d_vocab, d_model)
        self.pos_embedding_table: nn.Embedding = nn.Embedding(d_time, d_model)

        self.trans_blocks = nn.ModuleList(
            [TransformerBlock(d_time, d_model, n_heads) for _ in range(4)]
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head: nn.Linear = nn.Linear(d_model, d_vocab)

    @override
    def forward(
        self, batch: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, d_time = batch.shape
        token_embd = self.tok_embedding_table(batch)  # (d_batch, d_time, d_model)
        pos_embd = self.pos_embedding_table(torch.arange(d_time))  # (d_time, d_model)
        x = token_embd + pos_embd

        for trans_block in self.trans_blocks:
            x = trans_block(x)

        x = self.layer_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            d_batch, d_time, d_vocab = logits.shape
            targets = targets.view(d_batch * d_time)
            logits = logits.view(d_batch * d_time, d_vocab)
            loss = F.cross_entropy(logits, targets)

        return (logits, loss)

    def generate(self, ctx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        was_training = self.training
        _ = self.eval()
        for _ in range(max_tokens):
            # Position table only covers [0, d_time); never feed a longer window
            ctx_cond = ctx[:, -d_time:]
            logits, _ = self(ctx_cond)
            logits = logits[:, -1, :]  # Isolate the last time dimension -> (d_batch, d_vocab)
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            ctx = torch.cat((ctx, next_token), dim=1)

        if was_training:
            _ = self.train()
        return ctx

    @torch.no_grad
    def evaluate_loss(self, data_split: DataSplit) -> EvalSplit:
        _ = self.eval()
        eval_split = EvalSplit()

        # Train eval
        losses = torch.zeros(eval_iters)
        for step in range(eval_iters):
            xb, yb = generate_batch(data_split.train, d_batch, d_time)
            _, loss = self(xb, yb)
            losses[step] = loss
        eval_split.train_loss = losses.mean(dim=0).item()

        # Val eval
        losses = torch.zeros(eval_iters)
        for step in range(eval_iters):
            xb, yb = generate_batch(data_split.val, d_batch, d_time)
            _, loss = self(xb, yb)
            losses[step] = loss
        eval_split.val_loss = losses.mean(dim=0).item()

        _ = self.train()

        return eval_split


def train() -> BigramLanguageModel:
    data_split = extract_data()
    bigram_model = BigramLanguageModel()

    optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=lr)
    prev_val_loss = float("inf")
    for step in range(training_steps):
        xb, yb = generate_batch(data_split.train, d_batch, d_time)
        _, loss = bigram_model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            eval_split = bigram_model.evaluate_loss(data_split)
            print(
                f"Iterations: {step}\tTraining loss = {eval_split.train_loss}\tValidation loss = {eval_split.val_loss}\n"
            )
            if eval_split.val_loss > prev_val_loss:
                break
            prev_val_loss = eval_split.val_loss

    return bigram_model


if __name__ == "__main__":
    bigram_model = train()

    init_ctx = torch.zeros((1, 1), dtype=torch.long)
    print(tokenizer.decode_stream(bigram_model.generate(init_ctx, max_tokens=500)))
