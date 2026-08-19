import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from helpers import DataSplit
from train import (
    TrainConfig,
    TrialResult,
    context_fits,
    load_data,
    train,
)

# --- Search bounds -----------------------------------------------------------
# Staged search: lr -> d_model -> d_time -> dropout.
# Edit these defaults, or override with the matching CLI flags.
# Explicit --*-values lists override the min/max/n grids when passed.


@dataclass(slots=True)
class SearchSpace:
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    lr_n: int = 5
    lr_values: tuple[float, ...] | None = None

    d_model_min: int = 64
    d_model_max: int = 384
    d_model_values: tuple[int, ...] | None = None
    d_head: int = 32

    d_time_min: int = 64
    d_time_max: int = 256
    d_time_values: tuple[int, ...] | None = None

    dropout_min: float = 0.0
    dropout_max: float = 0.2
    dropout_n: int = 3
    dropout_values: tuple[float, ...] | None = None

    d_model_start: int = 128
    d_time_start: int = 64
    dropout_start: float = 0.2
    d_training_batch: int = 32
    n_layers: int = 4
    hold_tokens_per_step: bool = True

    lr_steps: int = 2000
    stage_steps: int = 8000
    eval_iters: int = 64
    early_stop_patience: int = 2
    seed: int = 1337


def geometric_ints(lo: int, hi: int) -> list[int]:
    if hi < lo:
        raise ValueError(f"max {hi} is below min {lo}")
    values: list[int] = []
    v = lo
    while v <= hi:
        values.append(v)
        nxt = v * 2
        if nxt <= v:
            break
        v = nxt
    if hi not in values:
        values.append(hi)
    return values


def closed_grid(lo: float, hi: float, n: int, *, log: bool) -> list[float]:
    if hi < lo:
        raise ValueError(f"max {hi} is below min {lo}")
    if n <= 1 or lo == hi:
        return [float(lo)]
    if log:
        if lo <= 0:
            raise ValueError("log grid requires min > 0")
        log_lo = math.log10(lo)
        log_hi = math.log10(hi)
        return [10 ** (log_lo + i * (log_hi - log_lo) / (n - 1)) for i in range(n)]
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def lr_grid(space: SearchSpace) -> list[float]:
    if space.lr_values is not None:
        return list(space.lr_values)
    return closed_grid(space.lr_min, space.lr_max, space.lr_n, log=True)


def dropout_grid(space: SearchSpace) -> list[float]:
    if space.dropout_values is not None:
        return list(space.dropout_values)
    return closed_grid(space.dropout_min, space.dropout_max, space.dropout_n, log=False)


def d_time_grid(space: SearchSpace) -> list[int]:
    if space.d_time_values is not None:
        return list(space.d_time_values)
    return geometric_ints(space.d_time_min, space.d_time_max)


def d_model_grid(space: SearchSpace) -> list[int]:
    if space.d_model_values is not None:
        raw = list(space.d_model_values)
    else:
        raw = geometric_ints(space.d_model_min, space.d_model_max)
    values = [v for v in raw if v >= space.d_head and v % space.d_head == 0]
    return values


def scaled_batch(base_batch: int, base_time: int, d_time: int) -> int:
    return max(1, (base_batch * base_time) // d_time)


def with_budget(cfg: TrainConfig, steps: int, space: SearchSpace) -> TrainConfig:
    return replace(
        cfg,
        training_steps=steps,
        eval_interval=max(1, steps // 10),
        eval_iters=space.eval_iters,
        early_stop_patience=space.early_stop_patience,
        seed=space.seed,
        verbose=True,
    )


def base_config(space: SearchSpace) -> TrainConfig:
    if space.d_model_start % space.d_head != 0:
        raise ValueError(
            f"d_model_start={space.d_model_start} is not divisible by d_head={space.d_head}"
        )
    return TrainConfig(
        d_time=space.d_time_start,
        d_training_batch=space.d_training_batch,
        d_model=space.d_model_start,
        n_heads=space.d_model_start // space.d_head,
        n_layers=space.n_layers,
        lr=1e-3,
        dropout=space.dropout_start,
        eval_iters=space.eval_iters,
        early_stop_patience=space.early_stop_patience,
        seed=space.seed,
    )


def search_skip_reason(cfg: TrainConfig, data_split: DataSplit) -> str | None:
    reason = cfg.illegal_reason() or context_fits(cfg, data_split)
    if reason is not None:
        return reason
    if cfg.dropout >= 0.3 and cfg.d_model <= 128:
        return f"dropout={cfg.dropout} >= 0.3 on d_model={cfg.d_model} <= 128"
    return None


def validate_space(space: SearchSpace) -> None:
    if not lr_grid(space):
        raise ValueError("lr grid is empty")
    if not d_model_grid(space):
        raise ValueError(
            f"no d_model in [{space.d_model_min}, {space.d_model_max}] "
            f"divisible by d_head={space.d_head}"
        )
    if not d_time_grid(space):
        raise ValueError("d_time grid is empty")
    if not dropout_grid(space):
        raise ValueError("dropout grid is empty")
    _ = base_config(space)


def format_cfg(cfg: TrainConfig) -> str:
    d_head = cfg.d_model // cfg.n_heads
    return (
        f"lr={cfg.lr:.4g} d_model={cfg.d_model} n_heads={cfg.n_heads} "
        f"(d_head={d_head}) d_time={cfg.d_time} d_training_batch={cfg.d_training_batch} "
        f"dropout={cfg.dropout:.3g} steps={cfg.training_steps}"
    )


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def pick_best(results: list[TrialResult]) -> TrialResult:
    if not results:
        raise RuntimeError("no completed trials to choose from")
    return min(results, key=lambda result: result.best_val_loss)


def save_winner(best: TrialResult, jsonl_path: Path, best_path: Path) -> TrainConfig:
    run_cfg = best.config.for_full_run()
    extra = {
        "best_val_loss": finite_or_none(best.best_val_loss),
        "best_train_loss": finite_or_none(best.best_train_loss),
        "search_steps": best.config.training_steps,
        "search_best_step": best.best_step,
        "diverged": best.diverged,
    }
    run_cfg.save(best_path, extra=extra)
    record = best.as_dict()
    record["stage"] = "winner"
    record["winner"] = True
    record["skipped"] = False
    record["config"] = run_cfg.to_json_dict()
    record["best_val_loss"] = finite_or_none(best.best_val_loss)
    record["best_train_loss"] = finite_or_none(best.best_train_loss)
    append_jsonl(jsonl_path, record)
    return run_cfg


def run_trial(
    cfg: TrainConfig,
    data_split: DataSplit,
    d_vocab: int,
    out_path: Path,
    stage: str,
) -> TrialResult | None:
    skip = search_skip_reason(cfg, data_split)
    if skip is not None:
        print(f"skip [{stage}] {skip} ({format_cfg(cfg)})")
        append_jsonl(
            out_path,
            {
                "stage": stage,
                "skipped": True,
                "reason": skip,
                "config": asdict(cfg),
            },
        )
        return None

    print(f"\n--- [{stage}] {format_cfg(cfg)} ---")
    _, result = train(cfg, data_split, d_vocab)
    record = result.as_dict()
    record["stage"] = stage
    record["skipped"] = False
    record["best_val_loss"] = finite_or_none(result.best_val_loss)
    record["best_train_loss"] = finite_or_none(result.best_train_loss)
    append_jsonl(out_path, record)
    val = "inf" if not math.isfinite(result.best_val_loss) else f"{result.best_val_loss:.4f}"
    print(
        f"    val={val} best_step={result.best_step} "
        f"steps_run={result.steps_run} time={result.wall_time_s:.1f}s "
        f"diverged={result.diverged}"
    )
    return result


def print_space(space: SearchSpace) -> None:
    print("Search space")
    print(f"  lr:      {['{:.4g}'.format(v) for v in lr_grid(space)]}")
    print(f"  d_model: {d_model_grid(space)}  (n_heads = d_model / {space.d_head})")
    print(f"  d_time:  {d_time_grid(space)}")
    print(f"  dropout: {['{:.3g}'.format(v) for v in dropout_grid(space)]}")
    print(
        f"  start:   d_model={space.d_model_start} d_time={space.d_time_start} "
        f"dropout={space.dropout_start} d_training_batch={space.d_training_batch}"
    )
    print(
        f"  budget:  lr_steps={space.lr_steps} stage_steps={space.stage_steps} "
        f"eval_iters={space.eval_iters}"
    )
    if space.hold_tokens_per_step:
        print(
            f"  d_training_batch scales with d_time to hold "
            f"{space.d_training_batch * space.d_time_start} tokens/step"
        )


def apply_d_time(cfg: TrainConfig, space: SearchSpace, d_time: int) -> TrainConfig:
    d_training_batch = cfg.d_training_batch
    if space.hold_tokens_per_step:
        d_training_batch = scaled_batch(
            space.d_training_batch, space.d_time_start, d_time
        )
    return replace(cfg, d_time=d_time, d_training_batch=d_training_batch)


def staged_search(
    space: SearchSpace,
    data_split: DataSplit,
    d_vocab: int,
    out_path: Path,
) -> TrialResult:
    winner = base_config(space)

    print("\n== stage 1: lr ==")
    lr_results: list[TrialResult] = []
    for lr in lr_grid(space):
        cfg = with_budget(replace(winner, lr=lr), space.lr_steps, space)
        result = run_trial(cfg, data_split, d_vocab, out_path, "lr")
        if result is not None:
            lr_results.append(result)
    winner = pick_best(lr_results).config
    print(f"best lr={winner.lr:.4g} val={pick_best(lr_results).best_val_loss:.4f}")

    print("\n== stage 2: d_model ==")
    width_results: list[TrialResult] = []
    for d_model in d_model_grid(space):
        cfg = with_budget(
            replace(
                winner,
                d_model=d_model,
                n_heads=d_model // space.d_head,
            ),
            space.stage_steps,
            space,
        )
        result = run_trial(cfg, data_split, d_vocab, out_path, "d_model")
        if result is not None:
            width_results.append(result)
    winner = pick_best(width_results).config
    print(
        f"best d_model={winner.d_model} n_heads={winner.n_heads} "
        f"val={pick_best(width_results).best_val_loss:.4f}"
    )

    print("\n== stage 3: d_time ==")
    time_results: list[TrialResult] = []
    for d_time in d_time_grid(space):
        cfg = with_budget(apply_d_time(winner, space, d_time), space.stage_steps, space)
        result = run_trial(cfg, data_split, d_vocab, out_path, "d_time")
        if result is not None:
            time_results.append(result)
    winner = pick_best(time_results).config
    print(
        f"best d_time={winner.d_time} d_training_batch={winner.d_training_batch} "
        f"val={pick_best(time_results).best_val_loss:.4f}"
    )

    print("\n== stage 4: dropout ==")
    drop_results: list[TrialResult] = []
    for dropout in dropout_grid(space):
        cfg = with_budget(replace(winner, dropout=dropout), space.stage_steps, space)
        result = run_trial(cfg, data_split, d_vocab, out_path, "dropout")
        if result is not None:
            drop_results.append(result)
    best = pick_best(drop_results)
    print(f"best dropout={best.config.dropout:.3g} val={best.best_val_loss:.4f}")
    return best


def sample_config(space: SearchSpace, rng: random.Random) -> TrainConfig:
    d_model = rng.choice(d_model_grid(space))
    d_time = rng.choice(d_time_grid(space))
    cfg = replace(
        base_config(space),
        d_model=d_model,
        n_heads=d_model // space.d_head,
        d_time=d_time,
        lr=rng.choice(lr_grid(space)),
        dropout=rng.choice(dropout_grid(space)),
    )
    return with_budget(apply_d_time(cfg, space, d_time), space.stage_steps, space)


def random_search(
    space: SearchSpace,
    data_split: DataSplit,
    d_vocab: int,
    out_path: Path,
    n_trials: int,
) -> TrialResult:
    rng = random.Random(space.seed)
    results: list[TrialResult] = []
    attempts = 0
    max_attempts = max(n_trials * 20, n_trials)
    while len(results) < n_trials and attempts < max_attempts:
        attempts += 1
        cfg = sample_config(space, rng)
        if search_skip_reason(cfg, data_split) is not None:
            continue
        result = run_trial(cfg, data_split, d_vocab, out_path, "random")
        if result is None:
            continue
        results.append(result)
    if len(results) < n_trials:
        print(
            f"warning: completed {len(results)}/{n_trials} random trials "
            f"({attempts} sample attempts)"
        )
    return pick_best(results)


def _parse_csv_floats(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_csv_ints(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_args() -> argparse.Namespace:
    defaults = SearchSpace()
    parser = argparse.ArgumentParser(
        description="Hyperparameter search: staged (lr -> d_model -> d_time -> dropout) or random."
    )
    parser.add_argument("--mode", choices=("staged", "random"), default="staged")
    parser.add_argument("--n-trials", type=int, default=20, help="Random mode only.")
    parser.add_argument("--out", type=Path, default=Path("sweep_results.jsonl"))
    parser.add_argument(
        "--best-out",
        type=Path,
        default=Path("best.json"),
        help="Write the winning TrainConfig here for `python train.py --config`.",
    )

    parser.add_argument("--lr-min", type=float, default=defaults.lr_min)
    parser.add_argument("--lr-max", type=float, default=defaults.lr_max)
    parser.add_argument("--lr-n", type=int, default=defaults.lr_n)
    parser.add_argument("--lr-values", type=str, default=None, help="e.g. 1e-4,3e-4,1e-3")

    parser.add_argument("--d-model-min", type=int, default=defaults.d_model_min)
    parser.add_argument("--d-model-max", type=int, default=defaults.d_model_max)
    parser.add_argument("--d-model-values", type=str, default=None, help="e.g. 64,128,256,384")
    parser.add_argument("--d-head", type=int, default=defaults.d_head)
    parser.add_argument("--d-model-start", type=int, default=defaults.d_model_start)

    parser.add_argument("--d-time-min", type=int, default=defaults.d_time_min)
    parser.add_argument("--d-time-max", type=int, default=defaults.d_time_max)
    parser.add_argument("--d-time-values", type=str, default=None, help="e.g. 64,128,256")
    parser.add_argument("--d-time-start", type=int, default=defaults.d_time_start)

    parser.add_argument("--dropout-min", type=float, default=defaults.dropout_min)
    parser.add_argument("--dropout-max", type=float, default=defaults.dropout_max)
    parser.add_argument("--dropout-n", type=int, default=defaults.dropout_n)
    parser.add_argument("--dropout-values", type=str, default=None, help="e.g. 0,0.1,0.2")
    parser.add_argument("--dropout-start", type=float, default=defaults.dropout_start)

    parser.add_argument("--d-training-batch", type=int, default=defaults.d_training_batch)
    parser.add_argument("--n-layers", type=int, default=defaults.n_layers)
    parser.add_argument(
        "--hold-tokens-per-step",
        action=argparse.BooleanOptionalAction,
        default=defaults.hold_tokens_per_step,
    )

    parser.add_argument("--lr-steps", type=int, default=defaults.lr_steps)
    parser.add_argument("--stage-steps", type=int, default=defaults.stage_steps)
    parser.add_argument("--eval-iters", type=int, default=defaults.eval_iters)
    parser.add_argument(
        "--early-stop-patience", type=int, default=defaults.early_stop_patience
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args()


def space_from_args(args: argparse.Namespace) -> SearchSpace:
    return SearchSpace(
        lr_min=args.lr_min,
        lr_max=args.lr_max,
        lr_n=args.lr_n,
        lr_values=_parse_csv_floats(args.lr_values),
        d_model_min=args.d_model_min,
        d_model_max=args.d_model_max,
        d_model_values=_parse_csv_ints(args.d_model_values),
        d_head=args.d_head,
        d_time_min=args.d_time_min,
        d_time_max=args.d_time_max,
        d_time_values=_parse_csv_ints(args.d_time_values),
        dropout_min=args.dropout_min,
        dropout_max=args.dropout_max,
        dropout_n=args.dropout_n,
        dropout_values=_parse_csv_floats(args.dropout_values),
        d_model_start=args.d_model_start,
        d_time_start=args.d_time_start,
        dropout_start=args.dropout_start,
        d_training_batch=args.d_training_batch,
        n_layers=args.n_layers,
        hold_tokens_per_step=args.hold_tokens_per_step,
        lr_steps=args.lr_steps,
        stage_steps=args.stage_steps,
        eval_iters=args.eval_iters,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
    )


def main() -> None:
    args = _parse_args()
    space = space_from_args(args)
    validate_space(space)
    print_space(space)

    data_split, d_vocab, _tokenizer = load_data()
    print(
        f"data: train={len(data_split.train)} val={len(data_split.val)} "
        f"test={len(data_split.test)} vocab={d_vocab}"
    )

    if args.mode == "staged":
        best = staged_search(space, data_split, d_vocab, args.out)
    else:
        best = random_search(space, data_split, d_vocab, args.out, args.n_trials)

    run_cfg = save_winner(best, args.out, args.best_out)

    print("\n== winner ==")
    print(format_cfg(run_cfg))
    print(
        f"val={best.best_val_loss:.4f} train={best.best_train_loss:.4f} "
        f"best_step={best.best_step} tokens={best.tokens_seen} "
        f"diverged={best.diverged}"
    )
    print(f"trial log: {args.out}")
    print(f"best config: {args.best_out}")
    print(f"retrain: python train.py --config {args.best_out}")


if __name__ == "__main__":
    main()
