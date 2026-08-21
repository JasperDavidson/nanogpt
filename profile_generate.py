from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from checkpoint import dir_for_name, load_manifest
from train import load_checkpoint

MAX_TOKENS = 500
POWER2 = (32, 64, 128, 256)
DENSE = (50, 60, 70, 80, 90, 100)
OUT_DIR = Path("plots")


def first_prefill_token(d_time: int) -> int:
    # 1-token prompt: generated token d_time is the first prefill-only step.
    return d_time


def split_times(times: np.ndarray, d_time: int) -> tuple[np.ndarray, np.ndarray]:
    switch = first_prefill_token(d_time)
    kv = times[: switch - 1]
    prefill = times[switch - 1 :]
    return kv, prefill


def time_one(d_time: int) -> np.ndarray:
    model, _ = load_checkpoint(dir_for_name(f"d_time_{d_time}"))
    init_ctx = torch.zeros((model.cfg.d_decode_batch, 1), dtype=torch.long)
    _ = model.generate(init_ctx, max_tokens=1)
    times: list[float] = []
    _ = model.generate(init_ctx, max_tokens=MAX_TOKENS, times=times)
    if len(times) != MAX_TOKENS:
        raise RuntimeError(f"d_time={d_time} recorded {len(times)} tokens, expected {MAX_TOKENS}")
    print(
        f"S={d_time}: {sum(times):.2f}s total, "
        f"switch at token {first_prefill_token(d_time)}"
    )
    return np.array(times, dtype=np.float64)


def plot_cumulative(
    results: dict[int, np.ndarray], path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    tokens = np.arange(1, MAX_TOKENS + 1)
    for d_time, times in results.items():
        switch = first_prefill_token(d_time)
        cumulative = np.cumsum(times)
        (line,) = ax.plot(tokens, cumulative, label=f"S={d_time}")
        ax.plot(
            switch,
            cumulative[switch - 1],
            "o",
            color=line.get_color(),
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.6,
            zorder=5,
        )
        ax.axvline(switch, color=line.get_color(), linestyle="--", alpha=0.35, linewidth=1)
    ax.set_xlabel("tokens generated")
    ax.set_ylabel("time so far (s)")
    ax.set_title(title)
    ax.legend()
    ax.text(
        0.98,
        0.02,
        "circle / dashed line = first prefill-only token (token S)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_phase_bars(
    results: dict[int, np.ndarray], path: Path, title: str
) -> None:
    d_times = list(results)
    kv_means = []
    prefill_means = []
    for d_time in d_times:
        kv, prefill = split_times(results[d_time], d_time)
        kv_means.append(float(kv.mean()) if kv.size else float("nan"))
        prefill_means.append(float(prefill.mean()) if prefill.size else float("nan"))

    x = np.arange(len(d_times))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, kv_means, width, label="avg KV-cache inter-token")
    ax.bar(x + width / 2, prefill_means, width, label="avg prefill inter-token")
    ax.set_xticks(x, [f"S={s}\nswitch @ {s}" for s in d_times])
    ax.set_ylabel("mean time between tokens (s)")
    ax.set_title(title)
    ax.legend()
    ax.text(
        0.98,
        0.02,
        "KV = tokens 1..S-1; prefill = tokens S..500",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_val_loss(d_times: tuple[int, ...], path: Path) -> None:
    losses = []
    for d_time in d_times:
        metrics = load_manifest(dir_for_name(f"d_time_{d_time}")).get("metrics")
        if not isinstance(metrics, dict) or "best_val_loss" not in metrics:
            raise KeyError(f"d_time_{d_time} checkpoint has no metrics.best_val_loss")
        losses.append(float(metrics["best_val_loss"]))

    s = np.array(d_times, dtype=np.float64)
    y = np.array(losses, dtype=np.float64)
    slope, intercept = np.polyfit(s, y, 1)
    xs = np.linspace(s.min(), s.max(), 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(s, y, zorder=3, label="best val loss")
    ax.plot(
        xs,
        slope * xs + intercept,
        color="C1",
        label=f"fit: val = {slope:.5f}·S + {intercept:.3f}",
    )
    ax.set_xlabel("context window (S)")
    ax.set_ylabel("best val loss")
    ax.set_title("Val loss vs context window (30k-step runs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("val loss fit:", {int(a): b for a, b in zip(s, y)})
    print(f"slope={slope:.6f} intercept={intercept:.4f}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    power2 = {s: time_one(s) for s in POWER2}
    dense = {s: time_one(s) for s in DENSE}

    plot_cumulative(
        power2,
        OUT_DIR / "cumulative_s32_256.png",
        "Cumulative generate time (S=32, 64, 128, 256)",
    )
    plot_phase_bars(
        power2,
        OUT_DIR / "bars_s32_256.png",
        "KV vs prefill mean inter-token time (S=32, 64, 128, 256)",
    )
    plot_cumulative(
        dense,
        OUT_DIR / "cumulative_s50_100.png",
        "Cumulative generate time (S=50, 60, 70, 80, 90, 100)",
    )
    plot_phase_bars(
        dense,
        OUT_DIR / "bars_s50_100.png",
        "KV vs prefill mean inter-token time (S=50, 60, 70, 80, 90, 100)",
    )
    plot_val_loss(DENSE, OUT_DIR / "val_loss_s50_100.png")
    print(f"wrote plots to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
