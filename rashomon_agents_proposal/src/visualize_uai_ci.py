"""UAI 风格统计可视化（bootstrap 95% CI + 配对差异）。

核心改进：
- 用 bootstrap over seeds 得到每个 epoch 的 95% CI（替代 std band）
- 用共同 seeds 的配对差异（Δ 相对 Baseline）做 CI 与可视化
- 成本用每个 seed 的 llm_stats.json 汇总（并在权衡图中加误差）

用法：
python -m src.visualize_uai_ci \
  --baseline output/01-22-11-04_baseline_multiseed \
  --no_rag output/01-22-11-05_ablation_no_rag \
  --no_subj output/01-22-11-05_ablation_no_subjective_graph \
  --no_trust output/01-22-11-05_ablation_no_llm_trust \
  --outdir output/uai_figures_ci \
  --bootstrap 20000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _style_uai():
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


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _seed_from_dirname(seed_dir: Path) -> Optional[int]:
    # seed_42 -> 42
    name = seed_dir.name
    if not name.startswith("seed_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def load_seed_level_series(exp_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Load per-seed aggregate metrics and llm stats from an experiment directory."""
    seed_data: Dict[int, Dict[str, Any]] = {}
    for seed_dir in sorted(exp_dir.glob("seed_*")):
        seed = _seed_from_dirname(seed_dir)
        if seed is None:
            continue
        agg_path = seed_dir / "aggregate_metrics.json"
        llm_path = seed_dir / "llm_stats.json"
        if not agg_path.exists():
            continue
        agg = _read_json(agg_path)
        llm = _read_json(llm_path) if llm_path.exists() else {}
        seed_data[seed] = {"aggregate": agg, "llm": llm}
    return seed_data


def extract_epoch_metric(agg: Dict[str, Any], epoch: int, key: str) -> float:
    # aggregate_metrics.json keys may be int or str; handle both
    epochs = agg.get("epochs", {})
    if str(epoch) in epochs:
        return float(epochs[str(epoch)][key])
    if epoch in epochs:
        return float(epochs[epoch][key])
    raise KeyError(f"Missing epoch {epoch} in aggregate metrics")


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) for mean(x) with percentile bootstrap CI."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    if x.size == 1:
        return mean, mean, mean
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.mean(x[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


@dataclass
class Setting:
    key: str
    label: str
    color: str
    linestyle: str
    exp_dir: Path
    seed_data: Dict[int, Dict[str, Any]]

    @property
    def seeds(self) -> List[int]:
        return sorted(self.seed_data.keys())

    def epoch_values(self, epoch: int, metric_key: str) -> np.ndarray:
        vals = []
        for s in self.seeds:
            vals.append(extract_epoch_metric(self.seed_data[s]["aggregate"], epoch, metric_key))
        return np.array(vals, dtype=float)

    def seed_costs(self) -> np.ndarray:
        costs = []
        for s in self.seeds:
            costs.append(float(self.seed_data[s].get("llm", {}).get("estimated_cost_usd", 0.0)))
        return np.array(costs, dtype=float)

    def seed_calls(self) -> np.ndarray:
        calls = []
        for s in self.seeds:
            calls.append(float(self.seed_data[s].get("llm", {}).get("total_calls", 0.0)))
        return np.array(calls, dtype=float)


def plot_trajectories_ci(settings: List[Setting], outdir: Path, n_boot: int, seed: int = 0):
    _style_uai()
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Assume epoch count from first seed of first setting
    any_seed = settings[0].seeds[0]
    epochs_dict = settings[0].seed_data[any_seed]["aggregate"]["epochs"]
    T = len(epochs_dict)
    x = np.arange(1, T + 1)

    # 说明：legend 放到图上方，避免与 x 轴标签重叠
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), constrained_layout=False)

    # DPAE
    ax = axes[0]
    for s in settings:
        means, lo, hi = [], [], []
        for e in range(T):
            v = s.epoch_values(e, "mean_dpae")
            m, l, h = bootstrap_mean_ci(v, n_boot, rng)
            means.append(m)
            lo.append(l)
            hi.append(h)
        means = np.array(means)
        lo = np.array(lo)
        hi = np.array(hi)
        ax.plot(x, means, label=s.label, color=s.color, linestyle=s.linestyle, linewidth=2)
        ax.fill_between(x, lo, hi, color=s.color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Exam (epoch)")
    ax.set_ylabel("DPAE (1 - Spearman)")
    ax.set_title("Collective Misperception (bootstrap 95% CI)")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)

    # Spearman rho
    ax = axes[1]
    for s in settings:
        means, lo, hi = [], [], []
        for e in range(T):
            v = s.epoch_values(e, "mean_spearman")
            m, l, h = bootstrap_mean_ci(v, n_boot, rng)
            means.append(m)
            lo.append(l)
            hi.append(h)
        means = np.array(means)
        lo = np.array(lo)
        hi = np.array(hi)
        ax.plot(x, means, label=s.label, color=s.color, linestyle=s.linestyle, linewidth=2)
        ax.fill_between(x, lo, hi, color=s.color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Exam (epoch)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Agreement (bootstrap 95% CI)")
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

    pdf = outdir / "traj_dpae_rho_bootstrap_ci.pdf"
    png = outdir / "traj_dpae_rho_bootstrap_ci.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)


def paired_delta_ci(
    baseline: Setting, other: Setting, epoch: int, metric_key: str, n_boot: int, rng: np.random.Generator
) -> Tuple[float, float, float, List[int]]:
    common = sorted(set(baseline.seeds) & set(other.seeds))
    if not common:
        return float("nan"), float("nan"), float("nan"), []
    diffs = []
    for s in common:
        b = extract_epoch_metric(baseline.seed_data[s]["aggregate"], epoch, metric_key)
        o = extract_epoch_metric(other.seed_data[s]["aggregate"], epoch, metric_key)
        diffs.append(o - b)
    diffs = np.array(diffs, dtype=float)
    mean = float(np.mean(diffs))
    if diffs.size == 1:
        return mean, mean, mean, common
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boots = np.mean(diffs[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi), common


def plot_paired_deltas(settings: List[Setting], outdir: Path, n_boot: int, seed: int = 0):
    _style_uai()
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    baseline = next(s for s in settings if s.key == "baseline")
    others = [s for s in settings if s.key != "baseline"]

    # final epoch index from baseline
    any_seed = baseline.seeds[0]
    T = len(baseline.seed_data[any_seed]["aggregate"]["epochs"])
    last = T - 1

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), constrained_layout=True)

    # ΔDPAE (other - baseline)
    ax = axes[0]
    means, err_lo, err_hi, labels, colors = [], [], [], [], []
    for s in others:
        m, lo, hi, common = paired_delta_ci(baseline, s, last, "mean_dpae", n_boot, rng)
        means.append(m)
        err_lo.append(m - lo)
        err_hi.append(hi - m)
        labels.append(s.label)
        colors.append(s.color)
    x = np.arange(len(others))
    ax.bar(x, means, color=colors, alpha=0.9)
    ax.errorbar(x, means, yerr=[err_lo, err_hi], fmt="none", ecolor="black", elinewidth=1.2, capsize=4)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(f"ΔDPAE (Epoch {T}): other − baseline")
    ax.set_title("Paired differences (bootstrap 95% CI)\n(common seeds=42/43/44)")

    # ΔSpearman (other - baseline)
    ax = axes[1]
    means, err_lo, err_hi, labels, colors = [], [], [], [], []
    for s in others:
        m, lo, hi, common = paired_delta_ci(baseline, s, last, "mean_spearman", n_boot, rng)
        means.append(m)
        err_lo.append(m - lo)
        err_hi.append(hi - m)
        labels.append(s.label)
        colors.append(s.color)
    x = np.arange(len(others))
    ax.bar(x, means, color=colors, alpha=0.9)
    ax.errorbar(x, means, yerr=[err_lo, err_hi], fmt="none", ecolor="black", elinewidth=1.2, capsize=4)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(f"ΔSpearman ρ (Epoch {T}): other − baseline")
    ax.set_title("Paired differences (bootstrap 95% CI)\n(common seeds=42/43/44)")

    pdf = outdir / "paired_deltas_final_epoch.pdf"
    png = outdir / "paired_deltas_final_epoch.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)


def plot_cost_tradeoff_ci(settings: List[Setting], outdir: Path, n_boot: int, seed: int = 0):
    """Mean cost vs final DPAE with bootstrap CI (across seeds within each setting)."""
    _style_uai()
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    baseline = settings[0]
    any_seed = baseline.seeds[0]
    T = len(baseline.seed_data[any_seed]["aggregate"]["epochs"])
    last = T - 1

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)

    for s in settings:
        # dpae CI
        d = s.epoch_values(last, "mean_dpae")
        d_m, d_lo, d_hi = bootstrap_mean_ci(d, n_boot, rng)
        # cost CI
        c = s.seed_costs()
        c_m, c_lo, c_hi = bootstrap_mean_ci(c, n_boot, rng)

        ax.errorbar(
            c_m,
            d_m,
            xerr=[[c_m - c_lo], [c_hi - c_m]],
            yerr=[[d_m - d_lo], [d_hi - d_m]],
            fmt="o",
            color=s.color,
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            markersize=7,
        )
        ax.text(c_m * 1.03 + 1e-3, d_m, s.label, fontsize=10, va="center")

    ax.set_xscale("log")
    ax.set_xlabel("Cost per setting (USD, mean across seeds; log scale)")
    ax.set_ylabel(f"Final DPAE (Epoch {T}; mean across seeds)")
    ax.set_title("Cost–Performance (bootstrap 95% CI)")
    ax.grid(True, alpha=0.25)

    pdf = outdir / "cost_vs_dpae_bootstrap_ci.pdf"
    png = outdir / "cost_vs_dpae_bootstrap_ci.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="UAI 风格统计可视化：bootstrap CI + 配对差异")
    parser.add_argument("--baseline", required=True, help="Baseline 实验输出目录（含 seed_*/aggregate_metrics.json）")
    parser.add_argument("--no_rag", required=True, help="No-RAG 实验输出目录")
    parser.add_argument("--no_subj", required=True, help="No-Subjective-Graph 实验输出目录")
    parser.add_argument("--no_trust", required=True, help="No-LLM-Trust 实验输出目录")
    parser.add_argument("--outdir", default="output/uai_figures_ci", help="输出目录")
    parser.add_argument("--bootstrap", type=int, default=20000, help="bootstrap 次数")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    args = parser.parse_args()

    outdir = Path(args.outdir)

    baseline_dir = Path(args.baseline)
    no_rag_dir = Path(args.no_rag)
    no_subj_dir = Path(args.no_subj)
    no_trust_dir = Path(args.no_trust)

    baseline = Setting(
        key="baseline",
        label="Baseline",
        color="#1f77b4",
        linestyle="-",
        exp_dir=baseline_dir,
        seed_data=load_seed_level_series(baseline_dir),
    )
    no_rag = Setting(
        key="no_rag",
        label="No-RAG",
        color="#ff7f0e",
        linestyle="--",
        exp_dir=no_rag_dir,
        seed_data=load_seed_level_series(no_rag_dir),
    )
    no_subj = Setting(
        key="no_subj",
        label="No-SubjGraph",
        color="#2ca02c",
        linestyle="--",
        exp_dir=no_subj_dir,
        seed_data=load_seed_level_series(no_subj_dir),
    )
    no_trust = Setting(
        key="no_trust",
        label="No-LLM-Trust",
        color="#d62728",
        linestyle="--",
        exp_dir=no_trust_dir,
        seed_data=load_seed_level_series(no_trust_dir),
    )

    settings = [baseline, no_rag, no_subj, no_trust]
    for s in settings:
        if not s.seeds:
            raise SystemExit(f"No seeds found in {s.exp_dir}")

    plot_trajectories_ci(settings, outdir, n_boot=args.bootstrap, seed=args.seed)
    plot_paired_deltas(settings, outdir, n_boot=args.bootstrap, seed=args.seed)
    plot_cost_tradeoff_ci(settings, outdir, n_boot=args.bootstrap, seed=args.seed)

    print(f"[OK] Saved CI figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

