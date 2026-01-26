"""Multi-seed visualization with ablation comparison.

Generates:
- Metric trajectories across epochs (mean ± std bands)
- Final epoch ablation comparison (error bars)
- Performance-cost trade-off scatter plot (Pareto-like)

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


@dataclass
class Setting:
    key: str
    label: str
    summary: Dict[str, Any]
    color: str
    linestyle: str = "-"


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _epochs(summary: Dict[str, Any]) -> List[int]:
    # keys are strings "0".."5"
    return sorted(int(k) for k in summary.get("epochs", {}).keys())


def _series(summary: Dict[str, Any], field: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) arrays for a field across epochs."""
    eps = _epochs(summary)
    mean = np.array([summary["epochs"][str(e)][f"{field}_mean"] for e in eps], dtype=float)
    std = np.array([summary["epochs"][str(e)][f"{field}_std"] for e in eps], dtype=float)
    return mean, std


def _series_simple(summary: Dict[str, Any], field: str) -> np.ndarray:
    eps = _epochs(summary)
    return np.array([summary["epochs"][str(e)][field] for e in eps], dtype=float)


def _style_figures():
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_metric_trajectories(
    settings: List[Setting],
    outdir: Path,
):
    """Plot DPAE and Spearman trajectories with uncertainty bands."""
    _style_uai()
    outdir.mkdir(parents=True, exist_ok=True)

    # x axis: epoch 1..T
    T = len(_epochs(settings[0].summary))
    x = np.arange(1, T + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), constrained_layout=False)

    # Left: DPAE
    ax = axes[0]
    for s in settings:
        m = np.array([s.summary["epochs"][str(e)]["dpae_mean"] for e in range(T)], dtype=float)
        sd = np.array([s.summary["epochs"][str(e)]["dpae_std"] for e in range(T)], dtype=float)
        ax.plot(x, m, label=s.label, color=s.color, linestyle=s.linestyle, linewidth=2)
        ax.fill_between(x, m - sd, m + sd, color=s.color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Exam (epoch)")
    ax.set_ylabel("DPAE (1 - Spearman)")
    ax.set_title("Collective Misperception over Time")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)

    # Right: Spearman rho
    ax = axes[1]
    for s in settings:
        m = np.array([s.summary["epochs"][str(e)]["spearman_mean"] for e in range(T)], dtype=float)
        sd = np.array([s.summary["epochs"][str(e)]["spearman_std"] for e in range(T)], dtype=float)
        ax.plot(x, m, label=s.label, color=s.color, linestyle=s.linestyle, linewidth=2)
        ax.fill_between(x, m - sd, m + sd, color=s.color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Exam (epoch)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Agreement with Ground-truth Rank")
    ax.set_xticks(x)
    ax.set_ylim(0.82, 0.98)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    pdf = outdir / "traj_dpae_rho_multiseed.pdf"
    png = outdir / "traj_dpae_rho_multiseed.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)


def plot_final_epoch_bars(settings: List[Setting], outdir: Path):
    """Final epoch bar plot with error bars (DPAE, rho) and top-k."""
    _style_uai()
    outdir.mkdir(parents=True, exist_ok=True)

    T = len(_epochs(settings[0].summary))
    last = str(T - 1)

    labels = [s.label for s in settings]
    x = np.arange(len(settings))

    dpae = np.array([s.summary["epochs"][last]["dpae_mean"] for s in settings])
    dpae_sd = np.array([s.summary["epochs"][last]["dpae_std"] for s in settings])
    rho = np.array([s.summary["epochs"][last]["spearman_mean"] for s in settings])
    rho_sd = np.array([s.summary["epochs"][last]["spearman_std"] for s in settings])
    top3 = np.array([s.summary["epochs"][last]["top3_accuracy_mean"] for s in settings])
    top5 = np.array([s.summary["epochs"][last]["top5_accuracy_mean"] for s in settings])

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), constrained_layout=True)

    ax = axes[0]
    ax.bar(x, dpae, yerr=dpae_sd, capsize=4, color=[s.color for s in settings], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("DPAE (lower is better)")
    ax.set_title(f"Final Misperception (Epoch {T})")

    ax = axes[1]
    ax.bar(x, rho, yerr=rho_sd, capsize=4, color=[s.color for s in settings], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Spearman ρ (higher is better)")
    ax.set_ylim(0.82, 0.98)
    ax.set_title(f"Final Agreement (Epoch {T})")

    ax = axes[2]
    width = 0.38
    ax.bar(x - width / 2, top3, width=width, color="#2ecc71", alpha=0.9, label="Top-3")
    ax.bar(x + width / 2, top5, width=width, color="#f39c12", alpha=0.9, label="Top-5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 0.85)
    ax.set_title(f"Top-k Recognition (Epoch {T})")
    ax.legend(frameon=False, loc="upper right")

    pdf = outdir / "final_epoch_ablation_bars.pdf"
    png = outdir / "final_epoch_ablation_bars.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)


def plot_cost_tradeoff(settings: List[Setting], outdir: Path):
    """Scatter: cost vs final DPAE (size by calls)."""
    _style_uai()
    outdir.mkdir(parents=True, exist_ok=True)

    T = len(_epochs(settings[0].summary))
    last = str(T - 1)

    costs = np.array([s.summary["total_cost_usd"] for s in settings], dtype=float)
    calls = np.array([s.summary["total_llm_calls"] for s in settings], dtype=float)
    dpae = np.array([s.summary["epochs"][last]["dpae_mean"] for s in settings], dtype=float)

    # bubble size: sqrt(calls) scaled
    size = 200 * (np.sqrt(calls) / (np.sqrt(calls).max() + 1e-9))

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    for i, s in enumerate(settings):
        ax.scatter(costs[i], dpae[i], s=size[i], color=s.color, alpha=0.9, edgecolor="black", linewidth=0.6)
        ax.text(costs[i] * 1.01 + 1e-3, dpae[i], s.label, fontsize=10, va="center")

    ax.set_xlabel("Total cost (USD, across seeds)")
    ax.set_ylabel(f"Final DPAE (Epoch {T})")
    ax.set_title("Cost–Performance Trade-off")
    ax.grid(True, alpha=0.25)
    ax.set_xscale("log")

    pdf = outdir / "cost_vs_dpae_tradeoff.pdf"
    png = outdir / "cost_vs_dpae_tradeoff.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed and ablation comparison visualization")
    parser.add_argument("--baseline", required=True, help="Path to baseline multi_seed_summary.json")
    parser.add_argument("--no_rag", required=True, help="Path to No-RAG multi_seed_summary.json")
    parser.add_argument("--no_subj", required=True, help="Path to No-Subjective-Graph multi_seed_summary.json")
    parser.add_argument("--no_trust", required=True, help="Path to No-LLM-Trust multi_seed_summary.json")
    parser.add_argument("--outdir", default="output/figures", help="Output directory")
    args = parser.parse_args()

    baseline = _read_json(Path(args.baseline))
    no_rag = _read_json(Path(args.no_rag))
    no_subj = _read_json(Path(args.no_subj))
    no_trust = _read_json(Path(args.no_trust))

    settings = [
        Setting("baseline", "Baseline", baseline, color="#1f77b4", linestyle="-"),
        Setting("no_rag", "No-RAG", no_rag, color="#ff7f0e", linestyle="--"),
        Setting("no_subj", "No-SubjGraph", no_subj, color="#2ca02c", linestyle="--"),
        Setting("no_trust", "No-LLM-Trust", no_trust, color="#d62728", linestyle="--"),
    ]

    outdir = Path(args.outdir)
    plot_metric_trajectories(settings, outdir)
    plot_final_epoch_bars(settings, outdir)
    plot_cost_tradeoff(settings, outdir)

    print(f"[OK] Saved figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

