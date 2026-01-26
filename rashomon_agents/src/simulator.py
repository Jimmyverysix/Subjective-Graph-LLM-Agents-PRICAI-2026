"""Simulator main loop and evaluation metrics.

Key outputs:
- `EpochMetrics` for each class per epoch
- `aggregate_metrics.json` aggregated across classes
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
from scipy import stats as scipy_stats
from tqdm import tqdm

from .data_loader import (
    StudentRecord, ClassData, DatasetSplit, 
    load_and_filter, load_config, get_all_students
)
from .subjective_graph import (
    SubjectiveGraph, RashomonSet, 
    build_all_rashomon_sets, compute_graph_stats
)
from .rag import RAGEngine, RAGMode
from .agent import LLMAgent, LLMClient, EvaluationMessage, AgentState


@dataclass
class EpochMetrics:
    """Evaluation metrics for a single epoch."""
    epoch: int
    class_code: str
    
    dpae: float = 0.0  # 1 - Spearman(ĥR, R)
    spearman_rho: float = 0.0
    
    top_k_accuracy: Dict[int, float] = field(default_factory=dict)
    
    mean_uncertainty: float = 0.0
    uncertainty_variance: float = 0.0
    
    mean_mu: float = 0.5
    std_mu: float = 0.0
    
    true_score_var: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "epoch": self.epoch,
            "class_code": self.class_code,
            "dpae": self.dpae,
            "spearman_rho": self.spearman_rho,
            "top_k_accuracy": self.top_k_accuracy,
            "mean_uncertainty": self.mean_uncertainty,
            "uncertainty_variance": self.uncertainty_variance,
            "mean_mu": self.mean_mu,
            "std_mu": self.std_mu,
            "true_score_var": self.true_score_var,
        }


def compute_dpae(
    belief_ranks: List[int],
    true_ranks: List[int],
) -> Tuple[float, float]:
    """Compute DPAE (Distributed Perception Accuracy Error).
    
    DPAE(t) = 1 - ρ(R̂^{(t)}, R^{(t)})
    
    Returns:
        (dpae, spearman_rho)
    """
    if len(belief_ranks) < 2:
        return 0.0, 0.0
    
    rho, _ = scipy_stats.spearmanr(belief_ranks, true_ranks)
    if np.isnan(rho):
        rho = 0.0
    
    dpae = 1 - rho
    return dpae, rho


def compute_top_k_accuracy(
    belief_ranks: Dict[int, int],
    true_ranks: Dict[int, int],
    k: int,
) -> float:
    """Compute Top-k accuracy.
    
    Acc@k(t) = |Top_k(R̂) ∩ Top_k(R)| / k
    """
    if len(belief_ranks) < k or len(true_ranks) < k:
        return 0.0
    
    belief_top_k = set(sorted(belief_ranks.keys(), key=lambda x: belief_ranks[x])[:k])
    true_top_k = set(sorted(true_ranks.keys(), key=lambda x: true_ranks[x])[:k])
    
    return len(belief_top_k & true_top_k) / k


def compute_class_metrics(
    agents: Dict[int, LLMAgent],
    class_data: ClassData,
    epoch: int,
) -> EpochMetrics:
    """Compute class evaluation metrics."""
    metrics = EpochMetrics(epoch=epoch, class_code=class_data.class_code)
    
    all_mus: Dict[int, List[float]] = defaultdict(list)
    all_sigmas: List[float] = []
    
    for agent in agents.values():
        for target_id, belief in agent.state.beliefs.items():
            all_mus[target_id].append(belief.mu)
            all_sigmas.append(belief.sigma)
    
    mean_beliefs: Dict[int, float] = {}
    for target_id, mus in all_mus.items():
        mean_beliefs[target_id] = np.mean(mus)
    
    sorted_by_belief = sorted(mean_beliefs.keys(), key=lambda x: -mean_beliefs[x])
    belief_ranks = {sid: rank + 1 for rank, sid in enumerate(sorted_by_belief)}
    
    true_ranks: Dict[int, int] = {}
    for sid, student in class_data.students.items():
        if epoch < len(student.exam_class_ranks):
            rank = student.exam_class_ranks[epoch]
            if not np.isnan(rank):
                true_ranks[sid] = int(rank)
    
    common_ids = set(belief_ranks.keys()) & set(true_ranks.keys())
    if len(common_ids) >= 2:
        belief_list = [belief_ranks[sid] for sid in common_ids]
        true_list = [true_ranks[sid] for sid in common_ids]
        metrics.dpae, metrics.spearman_rho = compute_dpae(belief_list, true_list)
    
    for k in [3, 5, 10]:
        if len(common_ids) >= k:
            metrics.top_k_accuracy[k] = compute_top_k_accuracy(
                belief_ranks, true_ranks, k
            )
    
    if all_sigmas:
        metrics.mean_uncertainty = np.mean(all_sigmas)
        metrics.uncertainty_variance = np.var(all_sigmas)
    
    all_mu_values = list(mean_beliefs.values())
    if all_mu_values:
        metrics.mean_mu = np.mean(all_mu_values)
        metrics.std_mu = np.std(all_mu_values)
    
    true_scores = []
    for student in class_data.students.values():
        if epoch < len(student.exam_class_ranks):
            rank = student.exam_class_ranks[epoch]
            if not np.isnan(rank):
                true_scores.append(rank)
    if true_scores:
        metrics.true_score_var = np.var(true_scores)
    
    return metrics


# ============================================================
# Simulator
# ============================================================

@dataclass
class SimulationConfig:
    """Simulation configuration."""
    epochs: int = 6
    rounds_per_epoch: int = 3
    max_partners: int = 3
    batch_targets: int = 5
    use_llm_trust: bool = True
    use_llm_message: bool = True
    seed: int = 42


class ClassSimulator:
    """Simulator for a single class."""
    
    def __init__(
        self,
        class_data: ClassData,
        rashomon: RashomonSet,
        rag_engine: RAGEngine,
        llm_client: LLMClient,
        config: SimulationConfig,
    ):
        self.class_data = class_data
        self.rashomon = rashomon
        self.rag = rag_engine
        self.llm = llm_client
        self.config = config
        
        self.rng = np.random.default_rng(config.seed)
        
        self.agents: Dict[int, LLMAgent] = {}
        for sid, student in class_data.students.items():
            graph = rashomon.graphs[sid]
            self.agents[sid] = LLMAgent(
                student=student,
                subjective_graph=graph,
                rag_engine=rag_engine,
                llm_client=llm_client,
                class_data=class_data,
            )
        
        self.metrics_history: List[EpochMetrics] = []
        self.message_log: List[Dict] = []
    
    def run_epoch(self, epoch: int, verbose: bool = False) -> EpochMetrics:
        """Run a single epoch."""
        if verbose:
            print(f"\n--- Epoch {epoch + 1}/{self.config.epochs} ---")
        
        for agent in self.agents.values():
            agent.set_self_anchor(epoch)
            agent.state.current_epoch = epoch
        
        for round_ in range(self.config.rounds_per_epoch):
            if verbose:
                print(f"  Round {round_ + 1}/{self.config.rounds_per_epoch}")
            
            self._run_round(epoch, round_, verbose)
        
        metrics = compute_class_metrics(self.agents, self.class_data, epoch)
        self.metrics_history.append(metrics)
        
        if verbose:
            print(f"  DPAE: {metrics.dpae:.4f}, Spearman ρ: {metrics.spearman_rho:.4f}")
            print(f"  Mean Uncertainty: {metrics.mean_uncertainty:.4f}")
        
        return metrics
    
    def _run_round(self, epoch: int, round_: int, verbose: bool):
        """Run a single round of interactions (parallelized version)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_messages: List[Tuple[int, EvaluationMessage]] = []
        
        agent_tasks = []
        for agent in self.agents.values():
            agent.state.current_round = round_
            
            neighbors = list(agent.graph.edges.keys())
            if not neighbors:
                continue
            
            n_partners = min(len(neighbors), self.config.max_partners)
            partners = self.rng.choice(neighbors, n_partners, replace=False).tolist()
            
            all_targets = set()
            for partner_id in partners:
                partner_graph = self.rashomon.graphs.get(partner_id)
                if partner_graph:
                    targets = list(partner_graph.edges.keys())
                    targets.append(partner_id)
                    all_targets.update(targets)
            
            all_targets.discard(agent.agent_id)
            all_targets = list(all_targets)[:self.config.batch_targets]
            
            if not all_targets:
                continue
            
            agent_tasks.append((agent, all_targets, partners))
        
        def process_agent(task):
            agent, all_targets, partners = task
            evaluations = agent.generate_evaluations(
                all_targets, epoch, round_, use_llm_message=self.config.use_llm_message
            )
            return agent, evaluations, partners
        
        with ThreadPoolExecutor(max_workers=min(100, len(agent_tasks))) as executor:
            futures = [executor.submit(process_agent, task) for task in agent_tasks]
            
            for future in as_completed(futures):
                try:
                    agent, evaluations, partners = future.result()
                    for partner_id in partners:
                        for msg in evaluations:
                            all_messages.append((partner_id, msg))
                            self.message_log.append({
                                "epoch": epoch,
                                "round": round_,
                                "sender": msg.sender_id,
                                "receiver": partner_id,
                                "target": msg.target_id,
                                "ability_score": msg.ability_score,
                                "uncertainty": msg.uncertainty,
                            })
                except Exception as e:
                    pass
        
        for receiver_id, msg in all_messages:
            receiver = self.agents.get(receiver_id)
            if receiver is None:
                continue
            
            sender = self.class_data.students.get(msg.sender_id)
            if sender is None:
                continue
            
            receiver.receive_and_update(
                msg, sender, 
                use_llm_trust=self.config.use_llm_trust
            )
    
    def run(self, verbose: bool = True) -> List[EpochMetrics]:
        """Run complete simulation."""
        for epoch in range(self.config.epochs):
            self.run_epoch(epoch, verbose)
        
        return self.metrics_history


class MultiClassSimulator:
    """Multi-class parallel simulator."""
    
    def __init__(
        self,
        split: DatasetSplit,
        rashomon_sets: Dict[str, RashomonSet],
        llm_client: LLMClient,
        config: SimulationConfig,
        rag_mode: RAGMode = RAGMode.SCOPED,
    ):
        self.split = split
        self.rashomon_sets = rashomon_sets
        self.llm = llm_client
        self.config = config
        
        self.rag = RAGEngine(split, rashomon_sets, seed=config.seed, mode=rag_mode)
        
        self.class_simulators: Dict[str, ClassSimulator] = {}
        for class_code, class_data in split.full_temporal.items():
            rashomon = rashomon_sets[class_code]
            self.class_simulators[class_code] = ClassSimulator(
                class_data=class_data,
                rashomon=rashomon,
                rag_engine=self.rag,
                llm_client=llm_client,
                config=config,
            )
    
    def run(self, verbose: bool = True, parallel_classes: bool = True) -> Dict[str, List[EpochMetrics]]:

        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        if parallel_classes and len(self.class_simulators) > 1:
            def run_class(class_code: str, sim: ClassSimulator):
                metrics = sim.run(verbose=False)
                return class_code, metrics
            
            with ThreadPoolExecutor(max_workers=min(12, len(self.class_simulators))) as executor:
                futures = {
                    executor.submit(run_class, cc, sim): cc 
                    for cc, sim in self.class_simulators.items()
                }
                
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Simulating classes (parallel)",
                    disable=not verbose,
                ):
                    class_code, metrics = future.result()
                    results[class_code] = metrics
                    if verbose:
                        last_metric = metrics[-1] if metrics else None
                        if last_metric:
                            print(f"  Class {class_code} completed: DPAE={last_metric.dpae:.4f}, ρ={last_metric.spearman_rho:.4f}")
        else:
            for class_code, sim in tqdm(
                self.class_simulators.items(),
                desc="Simulating classes",
                disable=not verbose,
            ):
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Class {class_code} simulation started")
                    print(f"{'='*60}")
                
                metrics = sim.run(verbose=verbose)
                results[class_code] = metrics
        
        return results
    
    def get_aggregate_metrics(
        self, results: Dict[str, List[EpochMetrics]]
    ) -> Dict[str, Any]:

        aggregate = {
            "n_classes": len(results),
            "n_agents": sum(
                len(sim.agents) for sim in self.class_simulators.values()
            ),
            "epochs": {},
        }
        
        for epoch in range(self.config.epochs):
            epoch_data = []
            for class_code, metrics_list in results.items():
                if epoch < len(metrics_list):
                    epoch_data.append(metrics_list[epoch])
            
            if epoch_data:
                aggregate["epochs"][epoch] = {
                    "mean_dpae": np.mean([m.dpae for m in epoch_data]),
                    "std_dpae": np.std([m.dpae for m in epoch_data]),
                    "mean_spearman": np.mean([m.spearman_rho for m in epoch_data]),
                    "mean_uncertainty": np.mean([m.mean_uncertainty for m in epoch_data]),
                    "top_3_accuracy": np.mean([
                        m.top_k_accuracy.get(3, 0) for m in epoch_data
                    ]),
                    "top_5_accuracy": np.mean([
                        m.top_k_accuracy.get(5, 0) for m in epoch_data
                    ]),
                }
        
        return aggregate
    
    def save_results(
        self,
        results: Dict[str, List[EpochMetrics]],
        output_dir: Path,
    ):
        """Save results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for class_code, metrics_list in results.items():
            class_file = output_dir / f"class_{class_code}_metrics.json"
            with open(class_file, "w", encoding="utf-8") as f:
                json.dump(
                    [m.to_dict() for m in metrics_list],
                    f, indent=2, ensure_ascii=False
                )
        
        aggregate = self.get_aggregate_metrics(results)
        agg_file = output_dir / "aggregate_metrics.json"
        with open(agg_file, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2, ensure_ascii=False)
        
        llm_stats = self.llm.get_stats()
        llm_file = output_dir / "llm_stats.json"
        with open(llm_file, "w", encoding="utf-8") as f:
            json.dump(llm_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_dir}")


