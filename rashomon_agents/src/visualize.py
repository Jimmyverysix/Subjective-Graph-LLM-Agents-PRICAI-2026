"""Visualization module - generate figures required for paper."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_results(output_dir: Path) -> Dict[str, Any]:
    """Load experimental results."""
    aggregate_file = output_dir / "seed_42" / "aggregate_metrics.json"
    with open(aggregate_file, "r", encoding="utf-8") as f:
        aggregate = json.load(f)
    
    class_metrics = {}
    for i in range(1, 13):
        class_file = output_dir / "seed_42" / f"class_{i}_metrics.json"
        if class_file.exists():
            with open(class_file, "r", encoding="utf-8") as f:
                class_metrics[i] = json.load(f)
    
    return {
        "aggregate": aggregate,
        "classes": class_metrics,
    }


def plot_dpae_trend(results: Dict, save_path: Path):
    """Plot DPAE trend over epochs."""
    aggregate = results["aggregate"]
    epochs = sorted([int(k) for k in aggregate["epochs"].keys()])
    
    dpae_mean = [aggregate["epochs"][str(e)]["mean_dpae"] for e in epochs]
    dpae_std = [aggregate["epochs"][str(e)]["std_dpae"] for e in epochs]
    spearman = [aggregate["epochs"][str(e)]["mean_spearman"] for e in epochs]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = '#e74c3c'
    ax1.set_xlabel('Exam (Epoch)', fontsize=14)
    ax1.set_ylabel('DPAE (Error)', fontsize=14, color=color1)
    ax1.errorbar([e+1 for e in epochs], dpae_mean, yerr=dpae_std, 
                 fmt='o-', color=color1, linewidth=2, markersize=8, 
                 capsize=5, label='DPAE')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 0.2)
    
    ax2 = ax1.twinx()
    color2 = '#3498db'
    ax2.set_ylabel('Spearman ρ (Correlation)', fontsize=14, color=color2)
    ax2.plot([e+1 for e in epochs], spearman, 's--', color=color2, 
             linewidth=2, markersize=8, label='Spearman ρ')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.8, 1.0)
    
    plt.title('Epistemic Uncertainty Propagation Over Time\n(DPAE ↑ = Collective Misperception Increases)', fontsize=14)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    ax1.set_xticks([e+1 for e in epochs])
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_uncertainty_trend(results: Dict, save_path: Path):
    """Plot uncertainty and Top-k accuracy trends."""
    aggregate = results["aggregate"]
    epochs = sorted([int(k) for k in aggregate["epochs"].keys()])
    
    uncertainty = [aggregate["epochs"][str(e)]["mean_uncertainty"] for e in epochs]
    top3 = [aggregate["epochs"][str(e)]["top_3_accuracy"] for e in epochs]
    top5 = [aggregate["epochs"][str(e)]["top_5_accuracy"] for e in epochs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot([e+1 for e in epochs], uncertainty, 'o-', color='#9b59b6', 
             linewidth=2, markersize=8)
    ax1.set_xlabel('Exam (Epoch)', fontsize=12)
    ax1.set_ylabel('Mean Belief Uncertainty (σ)', fontsize=12)
    ax1.set_title('Belief Uncertainty Over Time\n(↓ = Beliefs Converging)', fontsize=12)
    ax1.set_ylim(0.2, 0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([e+1 for e in epochs])
    
    ax2.plot([e+1 for e in epochs], top3, 'o-', color='#27ae60', 
             linewidth=2, markersize=8, label='Top-3 Accuracy')
    ax2.plot([e+1 for e in epochs], top5, 's--', color='#f39c12', 
             linewidth=2, markersize=8, label='Top-5 Accuracy')
    ax2.set_xlabel('Exam (Epoch)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Top-k "Best Student" Recognition\n(↓ = Harder to Identify Top Performers)', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([e+1 for e in epochs])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_class_comparison(results: Dict, save_path: Path):
    """Plot final DPAE comparison across classes."""
    class_metrics = results["classes"]
    
    classes = sorted(class_metrics.keys())
    final_dpae = []
    for c in classes:
        last_epoch = class_metrics[c][-1]
        final_dpae.append(last_epoch["dpae"])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#e74c3c' if d > 0.15 else '#3498db' for d in final_dpae]
    bars = ax.bar([f'Class {c}' for c in classes], final_dpae, color=colors, edgecolor='black')
    
    ax.axhline(y=np.mean(final_dpae), color='black', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(final_dpae):.3f}')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Final DPAE (Epoch 6)', fontsize=12)
    ax.set_title('Class-wise Collective Misperception\n(Red = High Error, Blue = Low Error)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, dpae in zip(bars, final_dpae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{dpae:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_dpae_by_class_over_time(results: Dict, save_path: Path):
    """Plot DPAE over time for each class (line plot)."""
    class_metrics = results["classes"]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    
    for i, (class_id, metrics) in enumerate(sorted(class_metrics.items())):
        epochs = [m["epoch"] + 1 for m in metrics]
        dpae = [m["dpae"] for m in metrics]
        ax.plot(epochs, dpae, 'o-', color=colors[i], linewidth=1.5, 
                markersize=5, label=f'Class {class_id}', alpha=0.7)
    
    ax.set_xlabel('Exam (Epoch)', fontsize=12)
    ax.set_ylabel('DPAE', fontsize=12)
    ax.set_title('DPAE Trajectory by Class\n(All classes show increasing misperception)', fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_summary_table(results: Dict) -> str:
    """Generate summary table (LaTeX format)."""
    aggregate = results["aggregate"]
    epochs = sorted([int(k) for k in aggregate["epochs"].keys()])
    
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Simulation Results: Epistemic Uncertainty Propagation}",
        "\\begin{tabular}{ccccc}",
        "\\toprule",
        "Epoch & DPAE & Spearman $\\rho$ & Uncertainty & Top-3 Acc \\\\",
        "\\midrule",
    ]
    
    for e in epochs:
        data = aggregate["epochs"][str(e)]
        lines.append(
            f"{e+1} & {data['mean_dpae']:.4f} & {data['mean_spearman']:.4f} & "
            f"{data['mean_uncertainty']:.4f} & {data['top_3_accuracy']:.2f} \\\\"
        )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:results}",
        "\\end{table}",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate visualization figures")
    parser.add_argument("--output_dir", required=True, help="Experiment output directory")
    parser.add_argument("--save_dir", default=None, help="Figure save directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    save_dir = Path(args.save_dir) if args.save_dir else output_dir / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {output_dir}")
    results = load_results(output_dir)
    
    print(f"\nGenerating figures...")
    plot_dpae_trend(results, save_dir / "dpae_trend.png")
    plot_uncertainty_trend(results, save_dir / "uncertainty_trend.png")
    plot_class_comparison(results, save_dir / "class_comparison.png")
    plot_dpae_by_class_over_time(results, save_dir / "dpae_by_class.png")
    
    latex_table = generate_summary_table(results)
    table_file = save_dir / "results_table.tex"
    with open(table_file, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"Saved: {table_file}")
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total classes: {results['aggregate']['n_classes']}")
    print(f"Total agents: {results['aggregate']['n_agents']}")
    print(f"\nKey Finding: DPAE increased from "
          f"{results['aggregate']['epochs']['0']['mean_dpae']:.4f} to "
          f"{results['aggregate']['epochs']['5']['mean_dpae']:.4f} "
          f"({(results['aggregate']['epochs']['5']['mean_dpae'] / results['aggregate']['epochs']['0']['mean_dpae'] - 1) * 100:.1f}% increase)")
    print(f"\nAll figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
