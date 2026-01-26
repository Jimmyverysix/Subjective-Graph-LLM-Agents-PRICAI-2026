#!/usr/bin/env python3
"""Main entry point for Rashomon Set Agents.

Usage:
    # Dry-run (single class, single epoch)
    python -m src.main --config config.yaml --dry-run
    
    # Full simulation (auto-generates timestamped output folder)
    python -m src.main --config config.yaml
    
    # Multi-seed experiments
    python -m src.main --config config.yaml --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from .data_loader import load_and_filter, load_config
from .subjective_graph import (
    build_all_rashomon_sets,
    build_all_objective_friend_sets,
    compute_graph_stats,
)
from .rag import RAGEngine, RAGMode
from .agent import LLMClient
from .simulator import (
    SimulationConfig, ClassSimulator, MultiClassSimulator, EpochMetrics
)


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   Rashomon Set Agents: LLM-Driven Multi-Agent Simulation         ║
║   ──────────────────────────────────────────────────────────     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_single_seed(
    config: dict,
    sim_config: SimulationConfig,
    seed: int,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
    rag_mode: RAGMode = RAGMode.SCOPED,
    use_subjective_graph: bool = True,
) -> Dict[str, Any]:
    """Run simulation for a single seed."""
    sim_config.seed = seed
    
    csv_path = config["data"]["cleaned_csv"]
    split = load_and_filter(csv_path)
    
    if use_subjective_graph:
        worry_noise_scale = config["simulation"].get("worry_noise_scale", 1.0)
        rashomon_sets = build_all_rashomon_sets(split, worry_noise_scale, seed=seed)
    else:
        rashomon_sets = build_all_objective_friend_sets(split, seed=seed)
    
    api_key = os.environ.get(config["llm"]["api_key_env"])
    max_workers = config.get("llm", {}).get("max_workers", 100)
    llm = LLMClient(
        base_url=config["llm"]["base_url"],
        model=config["llm"]["model"],
        api_key=api_key,
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
        cache_dir=output_dir / "cache" if output_dir else None,
        max_workers=max_workers,
    )
    
    if dry_run:
        test_class = list(split.full_temporal.keys())[0]
        class_data = split.full_temporal[test_class]
        rashomon = rashomon_sets[test_class]
        
        sim = ClassSimulator(
            class_data=class_data,
            rashomon=rashomon,
            rag_engine=RAGEngine(
                split,
                rashomon_sets,
                top_k=config.get("rag", {}).get("top_k", 8),
                seed=seed,
                mode=rag_mode,
            ),
            llm_client=llm,
            config=sim_config,
        )
        
        start_time = time.time()
        metrics = sim.run(verbose=verbose)
        elapsed = time.time() - start_time
        
        return {
            "seed": seed,
            "elapsed_seconds": elapsed,
            "llm_stats": llm.get_stats(),
            "metrics": {test_class: [m.to_dict() for m in metrics]},
        }
    else:
        multi_sim = MultiClassSimulator(
            split=split,
            rashomon_sets=rashomon_sets,
            llm_client=llm,
            config=sim_config,
            rag_mode=rag_mode,
        )
        
        start_time = time.time()
        results = multi_sim.run(verbose=verbose)
        elapsed = time.time() - start_time
        
        seed_output = output_dir / f"seed_{seed}"
        multi_sim.save_results(results, seed_output)
        
        aggregate = multi_sim.get_aggregate_metrics(results)
        
        return {
            "seed": seed,
            "elapsed_seconds": elapsed,
            "llm_stats": llm.get_stats(),
            "aggregate": aggregate,
        }


def load_existing_seed_result(output_dir: Path, seed: int) -> Optional[Dict[str, Any]]:
    """Load existing seed result if available (for resuming)."""
    seed_dir = output_dir / f"seed_{seed}"
    agg_file = seed_dir / "aggregate_metrics.json"
    llm_file = seed_dir / "llm_stats.json"
    
    if not agg_file.exists():
        return None
    
    try:
        with open(agg_file, "r", encoding="utf-8") as f:
            aggregate = json.load(f)
        
        llm_stats = {"total_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0, 
                     "cache_hits": 0, "estimated_cost_usd": 0.0}
        if llm_file.exists():
            with open(llm_file, "r", encoding="utf-8") as f:
                llm_stats = json.load(f)
        
        return {
            "seed": seed,
            "elapsed_seconds": 0,
            "llm_stats": llm_stats,
            "aggregate": aggregate,
        }
    except (json.JSONDecodeError, IOError):
        return None


def run_multi_seed(
    config: dict,
    sim_config: SimulationConfig,
    seeds: List[int],
    output_dir: Path,
    verbose: bool = True,
    rag_mode: RAGMode = RAGMode.SCOPED,
    use_subjective_graph: bool = True,
) -> Dict[str, Any]:
    """Run multi-seed experiments (supports resuming: skips completed seeds)."""
    all_results = []
    
    for i, seed in enumerate(seeds):
        existing = load_existing_seed_result(output_dir, seed)
        if existing is not None:
            print(f"\n{'='*60}")
            print(f"Seed {seed} ({i+1}/{len(seeds)}) - Already completed, skipping")
            print(f"{'='*60}")
            all_results.append(existing)
            continue
        
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({i+1}/{len(seeds)})")
        print(f"{'='*60}")
        
        result = run_single_seed(
            config, sim_config, seed, output_dir, 
            dry_run=False,
            verbose=verbose,
            rag_mode=rag_mode,
            use_subjective_graph=use_subjective_graph,
        )
        all_results.append(result)
    
    summary = compute_cross_seed_stats(all_results, sim_config.epochs)
    
    summary_file = output_dir / "multi_seed_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary


def compute_cross_seed_stats(
    results: List[Dict],
    n_epochs: int,
) -> Dict[str, Any]:
    """Compute cross-seed statistics."""
    summary = {
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "total_elapsed_seconds": sum(r["elapsed_seconds"] for r in results),
        "total_llm_calls": sum(r["llm_stats"]["total_calls"] for r in results),
        "total_cost_usd": sum(r["llm_stats"]["estimated_cost_usd"] for r in results),
        "epochs": {},
    }
    
    for epoch in range(n_epochs):
        epoch_dpaes = []
        epoch_spearmans = []
        epoch_uncertainties = []
        epoch_top3 = []
        epoch_top5 = []
        
        for r in results:
            if "aggregate" in r and str(epoch) in r["aggregate"].get("epochs", {}):
                data = r["aggregate"]["epochs"][str(epoch)]
            elif "aggregate" in r and epoch in r["aggregate"].get("epochs", {}):
                data = r["aggregate"]["epochs"][epoch]
            else:
                continue
                
            epoch_dpaes.append(data.get("mean_dpae", 0))
            epoch_spearmans.append(data.get("mean_spearman", 0))
            epoch_uncertainties.append(data.get("mean_uncertainty", 0))
            epoch_top3.append(data.get("top_3_accuracy", 0))
            epoch_top5.append(data.get("top_5_accuracy", 0))
        
        if epoch_dpaes:
            summary["epochs"][epoch] = {
                "dpae_mean": np.mean(epoch_dpaes),
                "dpae_std": np.std(epoch_dpaes),
                "dpae_ci_95": (
                    np.percentile(epoch_dpaes, 2.5),
                    np.percentile(epoch_dpaes, 97.5),
                ) if len(epoch_dpaes) > 1 else (0, 0),
                "spearman_mean": np.mean(epoch_spearmans),
                "spearman_std": np.std(epoch_spearmans),
                "uncertainty_mean": np.mean(epoch_uncertainties),
                "top3_accuracy_mean": np.mean(epoch_top3),
                "top5_accuracy_mean": np.mean(epoch_top5),
            }
    
    return summary


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Rashomon Set Agents: LLM-Driven Multi-Agent Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run test
  python -m src.main --config config.yaml --dry-run
  
  # Full simulation (single seed)
  python -m src.main --config config.yaml --output output
  
  # Multi-seed experiments
  python -m src.main --config config.yaml --seeds 42,43,44,45,46 --output output_multiseed
  
  # Run single epoch (quick validation)
  python -m src.main --config config.yaml --epochs 1 --output output_quick
"""
    )
    
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--output", default=None, help="Output directory (default: timestamp)")
    parser.add_argument("--resume-dir", default=None, help="Resume: specify existing output directory (skip completed seeds)")
    parser.add_argument("--dry-run", action="store_true", help="Run single class, single epoch")
    parser.add_argument("--seeds", type=str, default="42", help="Random seeds (comma-separated)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs in config")
    parser.add_argument("--rounds", type=int, default=None, help="Override rounds per epoch in config")
    parser.add_argument("--workers", type=int, default=100, help="LLM parallel worker threads")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    parser.add_argument("--no-rag", action="store_true", help="Ablation: disable target-related retrieval (only self + class_stats)")
    parser.add_argument("--no-subjective-graph", action="store_true", help="Ablation: use objective friend graph (no inferred edges/no anxiety noise)")
    parser.add_argument("--no-llm-trust", action="store_true", help="Ablation: disable LLM trust gating (use consistency function)")
    parser.add_argument("--no-llm-message", action="store_true", help="Ablation: disable LLM message generation (use heuristic messages)")
    
    args = parser.parse_args()
    
    print_banner()
    
    config = load_config(args.config)
    
    config.setdefault("llm", {})["max_workers"] = args.workers
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    sim_config = SimulationConfig(
        epochs=args.epochs or config["simulation"]["epochs"],
        rounds_per_epoch=args.rounds or config["simulation"]["rounds_per_epoch"],
        max_partners=config["simulation"]["max_partners"],
        batch_targets=config["simulation"]["batch_targets"],
        use_llm_trust=(not args.no_llm_trust),
        use_llm_message=(not args.no_llm_message),
    )
    
    if args.dry_run:
        sim_config.epochs = 1
        sim_config.rounds_per_epoch = 1
    
    if args.resume_dir:
        output_dir = Path(args.resume_dir)
        if not output_dir.exists():
            print(f"Error: --resume-dir directory does not exist: {output_dir}")
            sys.exit(1)
        print(f"[Resume mode] Using existing output directory: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%m-%d-%H-%M")
        run_name = f"{timestamp}_{args.output}" if args.output else timestamp
        output_dir = Path("output") / run_name
    
    print("Configuration:")
    print(f"  Epochs: {sim_config.epochs}")
    print(f"  Rounds/Epoch: {sim_config.rounds_per_epoch}")
    print(f"  Max Partners: {sim_config.max_partners}")
    print(f"  Batch Targets: {sim_config.batch_targets}")
    print(f"  Seeds: {seeds}")
    print(f"  LLM Workers: {args.workers}")
    print(f"  Output: {output_dir}")
    print(f"  Dry-run: {args.dry_run}")
    print(
        f"  Ablations: no_rag={args.no_rag}, no_subjective_graph={args.no_subjective_graph}, "
        f"no_llm_trust={args.no_llm_trust}, no_llm_message={args.no_llm_message}"
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = output_dir / "config_used.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump({
            "simulation": {
                "epochs": sim_config.epochs,
                "rounds_per_epoch": sim_config.rounds_per_epoch,
                "max_partners": sim_config.max_partners,
                "batch_targets": sim_config.batch_targets,
            },
            "seeds": seeds,
            "dry_run": args.dry_run,
        }, f, indent=2)
    
    verbose = not args.quiet
    
    start_time = time.time()

    rag_mode = RAGMode.NO_RAG if args.no_rag else RAGMode.SCOPED
    use_subjective_graph = (not args.no_subjective_graph)
    
    if args.dry_run:
        result = run_single_seed(
            config, sim_config, seeds[0], output_dir,
            dry_run=True,
            verbose=verbose,
            rag_mode=rag_mode,
            use_subjective_graph=use_subjective_graph,
        )
        print(f"\nElapsed time: {result['elapsed_seconds']:.2f} seconds")
        print(f"LLM stats: {result['llm_stats']}")
    elif len(seeds) == 1:
        result = run_single_seed(
            config, sim_config, seeds[0], output_dir,
            dry_run=False,
            verbose=verbose,
            rag_mode=rag_mode,
            use_subjective_graph=use_subjective_graph,
        )
        
        print(f"\n{'='*60}")
        print("Run completed")
        print(f"{'='*60}")
        print(f"Elapsed time: {result['elapsed_seconds']:.2f} seconds")
        print(f"LLM stats: {result['llm_stats']}")
        
        if "aggregate" in result:
            print("\nAggregate metrics:")
            for epoch, data in result["aggregate"]["epochs"].items():
                print(f"  Epoch {int(epoch)+1}: DPAE={data['mean_dpae']:.4f}, "
                      f"ρ={data['mean_spearman']:.4f}")
    else:
        summary = run_multi_seed(
            config,
            sim_config,
            seeds,
            output_dir,
            verbose=verbose,
            rag_mode=rag_mode,
            use_subjective_graph=use_subjective_graph,
        )
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("Multi-seed experiments completed")
        print(f"{'='*60}")
        print(f"Total seeds: {summary['n_seeds']}")
        print(f"Total elapsed time: {total_time:.2f} seconds")
        print(f"Total LLM calls: {summary['total_llm_calls']}")
        print(f"Total estimated cost: ${summary['total_cost_usd']:.4f}")
        
        print("\nCross-seed statistics (mean ± std):")
        for epoch, data in summary["epochs"].items():
            print(f"  Epoch {int(epoch)+1}:")
            print(f"    DPAE: {data['dpae_mean']:.4f} ± {data['dpae_std']:.4f}")
            print(f"    Spearman ρ: {data['spearman_mean']:.4f} ± {data['spearman_std']:.4f}")
            print(f"    Top-3 Accuracy: {data['top3_accuracy_mean']:.4f}")
            print(f"    Top-5 Accuracy: {data['top5_accuracy_mean']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
