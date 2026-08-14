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
vocab_size = -1
time_dim = 64
batch_size = 32
model_dim = 128
training_steps = 100000
lr = 1e-3
eval_iters = training_steps // 10
eval_interval = training_steps // 10
num_heads = 4

tokenizer: Tokenizer = Tokenizer()


def extract_data() -> DataSplit:
    global vocab_size

    data = tokenizer.get_data_split(val_percentage=0.1, test_percentage=0.1)
    vocab_size = tokenizer.get_vocab_size()

    print(f"train shape: {len(data.train.shape)}")
    print(f"val shape: {len(data.val.shape)}")
    print(f"test shape: {len(data.test.shape)}")

    return data


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()

        self.head_dim = int(model_dim / num_heads)
        self.num_heads = num_heads

        self.layer_norm = nn.LayerNorm((model_dim))
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)

        self.output = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B = input.shape[0]
        T = input.shape[1]
        C = input.shape[2]

        ln_input = self.layer_norm(input)
        q = (
            self.query.forward(ln_input)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key.forward(ln_input)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value.forward(ln_input)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        affinity = (
            q @ k.transpose(-2, -1)
        )  # Note only transpose along (time, feature) dimension; attention is not cross-batch

        tril = torch.tril(torch.ones(self.num_heads, T, T))
        affinity = affinity.masked_fill(tril == 0, float("-inf"))
        affinity *= (
            1 / (self.head_dim**0.5)
        )  # Reduce the variance after head_dim ~mean=0, variance=1 elements accumulate through dot
        affinity = F.softmax(
            affinity, dim=-1
        )  # Only softmax across the feature dimension

        a_out = (affinity @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.output.forward(a_out)


class NormHidden(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()

        self.linear = nn.Linear(model_dim, model_dim, bias=False)
        self.layer_norm = nn.LayerNorm((model_dim))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ln_input = self.layer_norm(input)
        hidden_out = self.linear(ln_input)

        return hidden_out


class TransformerBlock(nn.Module):
    def __init__(self, time_dim: int, model_dim: int, num_heads: int):
        super().__init__()

        self.attn = SelfAttention(model_dim, num_heads)
        self.ffn = NormHidden(model_dim)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attention = self.attn(input) + input
        hidden = self.relu(self.ffn(attention)) + attention

        return hidden


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embedding_table: nn.Embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding_table: nn.Embedding = nn.Embedding(time_dim, model_dim)

        self.trans_blocks = nn.ModuleList(
            [TransformerBlock(time_dim, model_dim, num_heads) for _ in range(4)]
        )

        self.layer_norm = nn.LayerNorm(model_dim)
        self.lm_head: nn.Linear = nn.Linear(model_dim, vocab_size)

    @override
    def forward(
        self, batch: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = batch.shape
        token_embd = self.tok_embedding_table(batch)  # (B, T, C)
        pos_embd = self.pos_embedding_table(torch.arange(T))  # (T, C)
        x = token_embd + pos_embd

        for trans_block in self.trans_blocks:
            x = trans_block(x)

        x = self.layer_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            targets = targets.view(B * T)
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets)

        return (logits, loss)

    def generate(self, ctx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        for _ in range(max_tokens):
            # Position table only covers [0, time_size); never feed a longer window
            ctx_cond = ctx[:, -time_dim:]
            logits, _ = self(ctx_cond)
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
            xb, yb = generate_batch(data_split.train, batch_size, time_dim)
            _, loss = self(xb, yb)
            losses[step] = loss
        eval_split.train_loss = losses.mean(dim=0).item()

        # Val eval
        losses = torch.zeros(eval_iters)
        for step in range(eval_iters):
            xb, yb = generate_batch(data_split.val, batch_size, time_dim)
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
        xb, yb = generate_batch(data_split.train, batch_size, time_dim)
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
