"""Plot episode returns, fixed-state Q estimates, and TD targets from task2_dqn.py."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "dqn": "DQN",
    "double_dqn": "Double DQN (uniforme)",
    "ddqn_per": "Double DQN + PER",
}
COLORS = {"dqn": "#2563eb", "double_dqn": "#e76f00", "ddqn_per": "#16845b"}


def load_runs(path: Path) -> dict[str, dict[int, list[dict]]]:
    runs: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            runs[row["algorithm"]][int(row["seed"])].append(row)
    for alg in runs:
        for seed in runs[alg]:
            runs[alg][seed].sort(key=lambda row: int(row["episode"]))
    return runs


def finite_matrix(runs: dict[int, list[dict]], key: str, rolling: bool = False, window: int = 20) -> np.ndarray:
    curves = []
    for rows in runs.values():
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        if rolling:
            if len(values) < window:
                continue
            values = np.convolve(values, np.ones(window) / window, mode="valid")
        curves.append(values)
    if not curves:
        return np.empty((0, 0))
    length = max(map(len, curves))
    padded = np.full((len(curves), length), np.nan, dtype=np.float64)
    for index, curve in enumerate(curves):
        padded[index, : len(curve)] = curve
    return padded


def mean_and_spread(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(matrix, axis=0)
    count = np.isfinite(matrix).sum(axis=0)
    spread = np.zeros_like(mean)
    multiple = count > 1
    if multiple.any():
        spread[multiple] = np.nanstd(matrix[:, multiple], axis=0, ddof=1)
    return mean, spread


def plot_metric(runs: dict[str, dict[int, list[dict]]], key: str, title: str, ylabel: str, output: Path, rolling: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.5), layout="constrained")
    for algorithm in ("dqn", "double_dqn", "ddqn_per"):
        if algorithm not in runs:
            continue
        matrix = finite_matrix(runs[algorithm], key, rolling=rolling)
        if matrix.size == 0:
            continue
        mean, spread = mean_and_spread(matrix)
        x = np.arange(20, 20 + len(mean)) if rolling else np.arange(1, len(mean) + 1)
        ax.plot(x, mean, color=COLORS[algorithm], linewidth=2, label=LABELS[algorithm])
        count = np.isfinite(matrix).sum(axis=0)
        band = np.where(count > 1, spread, np.nan)
        ax.fill_between(x, mean - band, mean + band, color=COLORS[algorithm], alpha=0.14, linewidth=0)
    ax.set(title=title, xlabel="Episodio", ylabel=ylabel)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summarize(runs: dict[str, dict[int, list[dict]]]) -> None:
    print("Algoritmo | semillas | recompensa media últimos 20 ep | Q fijo último episodio | target último ep | primer cruce 0/100/200")
    for algorithm, seeds in runs.items():
        values = []
        for seed, rows in seeds.items():
            tail = rows[-20:]
            values.append(
                (
                    np.mean([float(row["episode_return"]) for row in tail]),
                    float(rows[-1]["q_eval_mean"]),
                    float(rows[-1]["target_mean"]),
                )
            )
        aggregate = np.asarray(values, dtype=float)
        means = aggregate.mean(axis=0)
        stds = aggregate.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(3)
        reward_curves = finite_matrix(seeds, "episode_return", rolling=True)
        reward_mean = np.nanmean(reward_curves, axis=0) if reward_curves.size else np.asarray([])
        crossings = []
        for threshold in (0, 100, 200):
            indexes = np.flatnonzero(reward_mean >= threshold)
            crossings.append(str(int(indexes[0] + 20)) if len(indexes) else "—")
        print(
            f"{LABELS[algorithm]} | {len(values)} | "
            f"{means[0]:.1f} ± {stds[0]:.1f} | {means[1]:.2f} ± {stds[1]:.2f} | "
            f"{means[2]:.2f} ± {stds[2]:.2f} | {' / '.join(crossings)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/task2_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.input)
    plot_metric(
        runs, "episode_return", "Recompensa por episodio (media móvil de 20 episodios)",
        "Recompensa media", args.output_dir / "fig_task2_rewards.png", rolling=True,
    )
    plot_metric(
        runs, "q_eval_mean", "Valor Q medio sobre los mismos 100 estados", "Media de maxₐ Q(s, a)",
        args.output_dir / "fig_task2_q_values.png",
    )
    plot_metric(
        runs, "target_mean", "Target de Bellman medio registrado por episodio", "Target medio",
        args.output_dir / "fig_task2_targets.png",
    )
    summarize(runs)
    print(f"Figures saved under {args.output_dir}")


if __name__ == "__main__":
    main()
