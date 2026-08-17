from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

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


def _coerce_config_value(expected: type, value: object) -> Any:
    if expected is bool:
        if not isinstance(value, bool):
            raise TypeError(f"expected bool, got {type(value).__name__}")
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TypeError(f"expected int, got {type(value).__name__}")
        return int(value)
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TypeError(f"expected float, got {type(value).__name__}")
        return float(value)
    return value


@dataclass(slots=True)
class TrainConfig:
    d_time: int = 64
    d_batch: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    lr: float = 1e-3
    dropout: float = 0.2
    training_steps: int = 100000
    eval_iters: int = 64
    eval_interval: int = 10000
    early_stop_patience: int = 2
    seed: int = 1337
    verbose: bool = True

    def illegal_reason(self) -> str | None:
        if self.d_time < 1:
            return "d_time < 1"
        if self.d_batch < 1:
            return "d_batch < 1"
        if self.d_model < 1:
            return "d_model < 1"
        if self.n_heads < 1:
            return "n_heads < 1"
        if self.n_layers < 1:
            return "n_layers < 1"
        if self.d_model % self.n_heads != 0:
            return f"d_model={self.d_model} not divisible by n_heads={self.n_heads}"
        d_head = self.d_model // self.n_heads
        if d_head < 16:
            return f"d_head={d_head} < 16"
        if not 0.0 <= self.dropout < 1.0:
            return f"dropout={self.dropout} outside [0, 1)"
        if self.lr <= 0:
            return f"lr={self.lr} <= 0"
        if self.training_steps < 1:
            return "training_steps < 1"
        if self.eval_iters < 1:
            return "eval_iters < 1"
        if self.eval_interval < 1:
            return "eval_interval < 1"
        return None

    def tokens_per_step(self) -> int:
        return self.d_batch * self.d_time

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("verbose", None)
        return data

    def for_full_run(self) -> TrainConfig:
        defaults = TrainConfig()
        return replace(
            self,
            training_steps=defaults.training_steps,
            eval_iters=defaults.eval_iters,
            eval_interval=defaults.eval_interval,
            early_stop_patience=defaults.early_stop_patience,
            verbose=True,
        )

    def save(self, path: Path, extra: dict[str, object] | None = None) -> None:
        payload: dict[str, object] = self.to_json_dict()
        if extra:
            payload["sweep"] = extra
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> TrainConfig:
        raw_obj: object = data.get("config", data)
        if not isinstance(raw_obj, dict):
            raise TypeError("config must be a JSON object")
        raw: dict[str, object] = raw_obj
        defaults = cls()
        updates: dict[str, Any] = {}
        for field in fields(cls):
            if field.name == "verbose" or field.name not in raw:
                continue
            updates[field.name] = _coerce_config_value(
                type(getattr(defaults, field.name)), raw[field.name]
            )
        return replace(defaults, **updates)

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        path = Path(path)
        if path.suffix == ".jsonl":
            winner: dict[str, object] | None = None
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if isinstance(record, dict) and record.get("winner") is True:
                        winner = record
            if winner is None:
                raise ValueError(
                    f"{path} has no record with winner=true. "
                    "Load best.json from the sweep instead of picking the lowest "
                    "val in the jsonl; stage budgets are not comparable."
                )
            return cls.from_mapping(winner)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        return cls.from_mapping(payload)


@dataclass(slots=True)
class TrialResult:
    config: TrainConfig
    best_val_loss: float
    best_train_loss: float
    best_step: int
    tokens_seen: int
    steps_run: int
    wall_time_s: float
    diverged: bool

    def as_dict(self) -> dict[str, object]:
        config = asdict(self.config)
        config.pop("verbose", None)
        return {
            "config": config,
            "best_val_loss": self.best_val_loss,
            "best_train_loss": self.best_train_loss,
            "best_step": self.best_step,
            "tokens_seen": self.tokens_seen,
            "steps_run": self.steps_run,
            "wall_time_s": self.wall_time_s,
            "diverged": self.diverged,
        }


def load_data(
    val_percentage: float = 0.1, test_percentage: float = 0.1
) -> tuple[DataSplit, int, Tokenizer]:
    tokenizer = Tokenizer()
    data_split = tokenizer.get_data_split(
        val_percentage=val_percentage, test_percentage=test_percentage
    )
    return data_split, tokenizer.get_vocab_size(), tokenizer


def context_fits(cfg: TrainConfig, data_split: DataSplit) -> str | None:
    train_len = len(data_split.train)
    val_len = len(data_split.val)
    if cfg.d_time >= train_len:
        return f"d_time={cfg.d_time} >= train length {train_len}"
    if cfg.d_time >= val_len:
        return f"d_time={cfg.d_time} >= val length {val_len}"
    return None


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()

        self.d_head = d_model // n_heads
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
    def __init__(self, d_model: int, dropout: float):
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
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()

        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.ffn = NormHidden(d_model, dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attention = self.attn(input) + input
        hidden = self.ffn(attention) + attention

        return hidden


class BigramLanguageModel(nn.Module):
    def __init__(self, cfg: TrainConfig, d_vocab: int):
        super().__init__()
        self.cfg = cfg
        self.tok_embedding_table: nn.Embedding = nn.Embedding(d_vocab, cfg.d_model)
        self.pos_embedding_table: nn.Embedding = nn.Embedding(cfg.d_time, cfg.d_model)

        self.trans_blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head: nn.Linear = nn.Linear(cfg.d_model, d_vocab)

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
        d_time = self.cfg.d_time
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
        cfg = self.cfg
        _ = self.eval()
        eval_split = EvalSplit()

        losses = torch.zeros(cfg.eval_iters)
        for step in range(cfg.eval_iters):
            xb, yb = generate_batch(data_split.train, cfg.d_batch, cfg.d_time)
            _, loss = self(xb, yb)
            losses[step] = loss
        eval_split.train_loss = losses.mean(dim=0).item()

        losses = torch.zeros(cfg.eval_iters)
        for step in range(cfg.eval_iters):
            xb, yb = generate_batch(data_split.val, cfg.d_batch, cfg.d_time)
            _, loss = self(xb, yb)
            losses[step] = loss
        eval_split.val_loss = losses.mean(dim=0).item()

        _ = self.train()

        return eval_split


def train(
    cfg: TrainConfig,
    data_split: DataSplit,
    d_vocab: int,
) -> tuple[BigramLanguageModel, TrialResult]:
    illegal = cfg.illegal_reason() or context_fits(cfg, data_split)
    if illegal is not None:
        raise ValueError(illegal)

    torch.manual_seed(cfg.seed)
    model = BigramLanguageModel(cfg, d_vocab)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_step = 0
    patience_left = cfg.early_stop_patience
    diverged = False
    steps_run = 0
    started = time.perf_counter()

    def consider_eval(step: int) -> bool:
        nonlocal best_val_loss, best_train_loss, best_step, patience_left, diverged
        eval_split = model.evaluate_loss(data_split)
        if cfg.verbose:
            print(
                f"Iterations: {step}\tTraining loss = {eval_split.train_loss}\tValidation loss = {eval_split.val_loss}\n"
            )
        if not math.isfinite(eval_split.val_loss):
            diverged = True
            return False
        if eval_split.val_loss < best_val_loss:
            best_val_loss = eval_split.val_loss
            best_train_loss = eval_split.train_loss
            best_step = step
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                return False
        return True

    for step in range(cfg.training_steps):
        xb, yb = generate_batch(data_split.train, cfg.d_batch, cfg.d_time)
        _, loss = model(xb, yb)
        if not torch.isfinite(loss):
            diverged = True
            steps_run = step + 1
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps_run = step + 1

        if step % cfg.eval_interval == 0:
            if not consider_eval(step):
                break
    else:
        last_step = cfg.training_steps - 1
        if last_step % cfg.eval_interval != 0:
            _ = consider_eval(last_step)

    tokens_seen = best_step * cfg.tokens_per_step()
    result = TrialResult(
        config=cfg,
        best_val_loss=best_val_loss,
        best_train_loss=best_train_loss,
        best_step=best_step,
        tokens_seen=tokens_seen,
        steps_run=steps_run,
        wall_time_s=time.perf_counter() - started,
        diverged=diverged,
    )
    return model, result


def _parse_train_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a nanoGPT run from a TrainConfig.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON from sweep (best.json) or JSONL with a winner=true record.",
    )
    parser.add_argument("--d-time", type=int, default=None)
    parser.add_argument("--d-batch", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--training-steps", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = TrainConfig.load(args.config) if args.config is not None else TrainConfig()
    updates: dict[str, Any] = {}
    for field in fields(TrainConfig):
        if field.name == "verbose":
            continue
        value = getattr(args, field.name, None)
        if value is not None:
            updates[field.name] = value
    return replace(cfg, **updates)


if __name__ == "__main__":
    cfg = _parse_train_args()
    data_split, d_vocab, tokenizer = load_data()
    bigram_model, result = train(cfg, data_split, d_vocab)
    print(
        f"best val={result.best_val_loss:.4f} at step {result.best_step} "
        f"(diverged={result.diverged})"
    )

    init_ctx = torch.zeros((1, 1), dtype=torch.long)
    print(tokenizer.decode_stream(bigram_model.generate(init_ctx, max_tokens=500)))
