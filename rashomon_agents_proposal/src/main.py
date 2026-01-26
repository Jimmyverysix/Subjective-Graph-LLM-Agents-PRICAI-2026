#!/usr/bin/env python3
"""Rashomon Set Agents 主入口。

用法:
    # Dry-run (仅一个班级，一个 epoch)
    python -m src.main --config config.yaml --dry-run
    
    # 完整模拟（自动生成时间戳输出文件夹）
    python -m src.main --config config.yaml
    
    # 多种子实验
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

# 加载 .env 文件
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
    """打印启动横幅。"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   Rashomon Set Agents: LLM-Driven Multi-Agent Simulation         ║
║   ──────────────────────────────────────────────────────────     ║
║   UAI 2026 Proposal Implementation                               ║
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
    """运行单个种子的模拟。"""
    sim_config.seed = seed
    
    # 加载数据
    csv_path = config["data"]["cleaned_csv"]
    split = load_and_filter(csv_path)
    
    # 构建 Rashomon 集合（支持消融：No-Subjective-Graph）
    if use_subjective_graph:
        worry_noise_scale = config["simulation"].get("worry_noise_scale", 1.0)
        rashomon_sets = build_all_rashomon_sets(split, worry_noise_scale, seed=seed)
    else:
        rashomon_sets = build_all_objective_friend_sets(split, seed=seed)
    
    # 创建 LLM 客户端（高并行度）
    api_key = os.environ.get(config["llm"]["api_key_env"])
    max_workers = config.get("llm", {}).get("max_workers", 100)  # 默认 100 并行
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
        # Dry-run: 只运行一个班级
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
        # 完整模拟
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
        
        # 保存结果
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
    """尝试加载已完成的 seed 结果（用于续跑）。"""
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
            "elapsed_seconds": 0,  # 已完成的不计时
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
    """运行多种子实验（支持续跑：跳过已完成的 seed）。"""
    all_results = []
    
    for i, seed in enumerate(seeds):
        # 检查是否已完成
        existing = load_existing_seed_result(output_dir, seed)
        if existing is not None:
            print(f"\n{'='*60}")
            print(f"种子 {seed} ({i+1}/{len(seeds)}) - 已完成，跳过")
            print(f"{'='*60}")
            all_results.append(existing)
            continue
        
        print(f"\n{'='*60}")
        print(f"种子 {seed} ({i+1}/{len(seeds)})")
        print(f"{'='*60}")
        
        result = run_single_seed(
            config, sim_config, seed, output_dir, 
            dry_run=False,
            verbose=verbose,
            rag_mode=rag_mode,
            use_subjective_graph=use_subjective_graph,
        )
        all_results.append(result)
    
    # 计算跨种子统计
    summary = compute_cross_seed_stats(all_results, sim_config.epochs)
    
    # 保存汇总
    summary_file = output_dir / "multi_seed_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary


def compute_cross_seed_stats(
    results: List[Dict],
    n_epochs: int,
) -> Dict[str, Any]:
    """计算跨种子的统计量。"""
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
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="Rashomon Set Agents: LLM-Driven Multi-Agent Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Dry-run 测试
  python -m src.main --config config.yaml --dry-run
  
  # 完整模拟（单种子）
  python -m src.main --config config.yaml --output output
  
  # 多种子实验
  python -m src.main --config config.yaml --seeds 42,43,44,45,46 --output output_multiseed
  
  # 只运行一个 epoch（快速验证）
  python -m src.main --config config.yaml --epochs 1 --output output_quick
"""
    )
    
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--output", default=None, help="输出目录（默认使用时间戳）")
    parser.add_argument("--resume-dir", default=None, help="续跑：指定已存在的输出目录（跳过已完成的 seed）")
    parser.add_argument("--dry-run", action="store_true", help="只运行一个班级一个 epoch")
    parser.add_argument("--seeds", type=str, default="42", help="随机种子（逗号分隔）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epoch 数")
    parser.add_argument("--rounds", type=int, default=None, help="覆盖配置中的每 epoch 轮数")
    parser.add_argument("--workers", type=int, default=100, help="LLM 并行线程数")
    parser.add_argument("--quiet", action="store_true", help="减少输出")
    parser.add_argument("--no-rag", action="store_true", help="消融：禁用目标相关检索（仅 self + class_stats）")
    parser.add_argument("--no-subjective-graph", action="store_true", help="消融：使用客观好友图（无推断边/无焦虑噪声）")
    parser.add_argument("--no-llm-trust", action="store_true", help="消融：禁用 LLM 信任门控（使用一致性函数）")
    parser.add_argument("--no-llm-message", action="store_true", help="消融：禁用 LLM 消息生成（使用启发式消息）")
    
    args = parser.parse_args()
    
    print_banner()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置并行线程数
    config.setdefault("llm", {})["max_workers"] = args.workers
    
    # 解析种子
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # 创建模拟配置
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
    
    # 生成输出目录：output/<时间戳>[_<tag>]/
    # 或使用 --resume-dir 续跑已存在的目录
    if args.resume_dir:
        output_dir = Path(args.resume_dir)
        if not output_dir.exists():
            print(f"错误：--resume-dir 指定的目录不存在: {output_dir}")
            sys.exit(1)
        print(f"[续跑模式] 使用已存在的输出目录: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%m-%d-%H-%M")
        run_name = f"{timestamp}_{args.output}" if args.output else timestamp
        output_dir = Path("output") / run_name
    
    # 打印配置
    print("配置:")
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
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
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
    
    # 运行模拟
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
        print(f"\n运行时间: {result['elapsed_seconds']:.2f} 秒")
        print(f"LLM 统计: {result['llm_stats']}")
    elif len(seeds) == 1:
        result = run_single_seed(
            config, sim_config, seeds[0], output_dir,
            dry_run=False,
            verbose=verbose,
            rag_mode=rag_mode,
            use_subjective_graph=use_subjective_graph,
        )
        
        print(f"\n{'='*60}")
        print("运行完成")
        print(f"{'='*60}")
        print(f"运行时间: {result['elapsed_seconds']:.2f} 秒")
        print(f"LLM 统计: {result['llm_stats']}")
        
        if "aggregate" in result:
            print("\n聚合指标:")
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
        print("多种子实验完成")
        print(f"{'='*60}")
        print(f"总种子数: {summary['n_seeds']}")
        print(f"总运行时间: {total_time:.2f} 秒")
        print(f"总 LLM 调用: {summary['total_llm_calls']}")
        print(f"总成本估计: ${summary['total_cost_usd']:.4f}")
        
        print("\n跨种子统计 (均值 ± 标准差):")
        for epoch, data in summary["epochs"].items():
            print(f"  Epoch {int(epoch)+1}:")
            print(f"    DPAE: {data['dpae_mean']:.4f} ± {data['dpae_std']:.4f}")
            print(f"    Spearman ρ: {data['spearman_mean']:.4f} ± {data['spearman_std']:.4f}")
            print(f"    Top-3 Accuracy: {data['top3_accuracy_mean']:.4f}")
            print(f"    Top-5 Accuracy: {data['top5_accuracy_mean']:.4f}")
    
    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
