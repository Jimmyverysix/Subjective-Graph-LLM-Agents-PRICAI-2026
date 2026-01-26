"""Subjective graph construction (Rashomon set).

Each student \(i\) has their own subjective graph \(G_i=(V,E_i,\pi_i)\), which constrains their visible neighborhood and information reachability.

Current implementation includes:
- **Explicit friend edges**: From `friend_1_id` ~ `friend_6_id`
- **Inferred edges (weak)**: From `most_popular_*_id` (as weak connections for "one-way attention")
- **Anxiety noise**: Missing/incorrect edges based on \(\alpha_i=g(worry\_about\_others)\)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import random

import numpy as np

from .data_loader import StudentRecord, ClassData, DatasetSplit


@dataclass
class SubjectiveEdge:
    """Edge in subjective graph."""
    source: int
    target: int
    weight: float = 1.0  # Subjective weight/credibility π_i(e) ∈ [0, 1]
    edge_type: str = "friend"  # Edge type: friend, inferred, noise
    
    def __hash__(self):
        return hash((self.source, self.target))
    
    def __eq__(self, other):
        if not isinstance(other, SubjectiveEdge):
            return False
        return self.source == other.source and self.target == other.target


@dataclass
class SubjectiveGraph:
    """Subjective graph for a single agent G_i = (V, E_i, π_i)."""
    owner_id: int
    class_code: str
    
    nodes: Set[int] = field(default_factory=set)
    
    edges: Dict[int, SubjectiveEdge] = field(default_factory=dict)  # target_id -> edge
    
    worry_level: int = 0  # Anxiety level (0-3)
    
    noise_alpha: float = 0.0  # α_i = g(worry_about_others)
    
    def get_neighbors(self) -> List[int]:
        """Get subjective neighbor list."""
        return list(self.edges.keys())
    
    def get_neighbor_weights(self) -> Dict[int, float]:
        """Get neighbors and their weights."""
        return {target: edge.weight for target, edge in self.edges.items()}
    
    def add_edge(self, target: int, weight: float = 1.0, edge_type: str = "friend"):
        """Add subjective edge."""
        if target == self.owner_id:
            return
        if target not in self.nodes:
            return
        self.edges[target] = SubjectiveEdge(
            source=self.owner_id,
            target=target,
            weight=weight,
            edge_type=edge_type,
        )
    
    def remove_edge(self, target: int):
        """Remove subjective edge."""
        if target in self.edges:
            del self.edges[target]
    
    @property
    def degree(self) -> int:
        """Subjective degree."""
        return len(self.edges)


@dataclass
class RashomonSet:
    """Rashomon set: collection of subjective graphs for all agents in a class.
    
    R = {G_1, ..., G_N}
    """
    class_code: str
    graphs: Dict[int, SubjectiveGraph] = field(default_factory=dict)  # student_id -> graph
    
    def __getitem__(self, student_id: int) -> SubjectiveGraph:
        return self.graphs[student_id]
    
    def __contains__(self, student_id: int) -> bool:
        return student_id in self.graphs
    
    @property
    def size(self) -> int:
        return len(self.graphs)


def worry_to_alpha(worry_level: int, scale: float = 1.0) -> float:
    """Map worry level to noise intensity α_i.
    
    Linear mapping: 0/0.33/0.66/1.0
    
    Args:
        worry_level: 0=never, 1=sometimes, 2=often, 3=very_concerned
        scale: Scaling factor (worry_noise_scale in config)
    
    Returns:
        α_i ∈ [0, 1]
    """
    base_alpha = worry_level / 3.0  # 0, 0.33, 0.67, 1.0
    return min(1.0, base_alpha * scale)


def apply_edge_noise(
    edges: Dict[int, SubjectiveEdge],
    all_classmates: Set[int],
    owner_id: int,
    alpha: float,
    rng: np.random.Generator,
    epsilon: float = 0.1,
) -> Dict[int, SubjectiveEdge]:
    """Apply anxiety noise to edge set.
    
    According to proposal formula:
    P((i,k) ∈ E_i) = (1 - α_i) · 1[k ∈ N_i^friend] + α_i · ε
    
    Args:
        edges: Original edge set
        all_classmates: All student IDs in class
        owner_id: Subjective graph owner ID
        alpha: Noise intensity
        rng: Random number generator
        epsilon: Small probability parameter for incorrect edges
    
    Returns:
        Edge set after noise injection
    """
    if alpha == 0:
        return edges
    
    noisy_edges = {}
    
    for target, edge in edges.items():
        if edge.edge_type == "friend":
            keep_prob = 1 - alpha
            if rng.random() < keep_prob:
                noisy_edges[target] = edge
        else:
            noisy_edges[target] = edge
    
    non_neighbors = all_classmates - {owner_id} - set(noisy_edges.keys())
    noise_prob = alpha * epsilon
    
    for candidate in non_neighbors:
        if rng.random() < noise_prob:
            noisy_edges[candidate] = SubjectiveEdge(
                source=owner_id,
                target=candidate,
                weight=0.3,
                edge_type="noise",
            )
    
    return noisy_edges


def build_subjective_graph(
    student: StudentRecord,
    class_data: ClassData,
    worry_noise_scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> SubjectiveGraph:
    """Build subjective graph for a single student.
    
    Args:
        student: Student record
        class_data: Class data
        worry_noise_scale: Anxiety noise scaling factor
        rng: Random number generator
    
    Returns:
        Subjective graph for this student
    """
    if rng is None:
        rng = np.random.default_rng()
    
    graph = SubjectiveGraph(
        owner_id=student.student_id,
        class_code=student.class_code,
        nodes=set(class_data.student_ids),
        worry_level=student.worry_level,
        noise_alpha=worry_to_alpha(student.worry_level, worry_noise_scale),
    )
    
    for friend_id in student.friend_ids:
        if friend_id in class_data.students:
            graph.add_edge(friend_id, weight=1.0, edge_type="friend")
    
    raw = student.raw_row
    
    for i in range(1, 4):
        col = f"most_popular_{i}_id"
        if col in raw and raw[col] is not None:
            try:
                pop_id = int(raw[col])
                if pop_id in class_data.students and pop_id != student.student_id:
                    if pop_id not in graph.edges:
                        graph.add_edge(pop_id, weight=0.5, edge_type="inferred")
            except (ValueError, TypeError):
                pass
    
    if graph.noise_alpha > 0:
        graph.edges = apply_edge_noise(
            edges=graph.edges,
            all_classmates=graph.nodes,
            owner_id=graph.owner_id,
            alpha=graph.noise_alpha,
            rng=rng,
        )
    
    return graph


def build_rashomon_set(
    class_data: ClassData,
    worry_noise_scale: float = 1.0,
    seed: int = 42,
) -> RashomonSet:
    """Build Rashomon set for entire class.
    
    Args:
        class_data: Class data
        worry_noise_scale: Anxiety noise scaling factor
        seed: Random seed
    
    Returns:
        Rashomon set for the class
    """
    rng = np.random.default_rng(seed)
    
    rashomon = RashomonSet(class_code=class_data.class_code)
    
    for student_id, student in class_data.students.items():
        graph = build_subjective_graph(
            student=student,
            class_data=class_data,
            worry_noise_scale=worry_noise_scale,
            rng=rng,
        )
        rashomon.graphs[student_id] = graph
    
    return rashomon


def build_objective_friend_rashomon_set(
    class_data: ClassData,
    seed: int = 42,
) -> RashomonSet:
    """消融：No-Subjective-Graph。
    
    所有智能体共享同一“客观好友图”生成规则：仅使用显式好友块构建邻接（对称化），
    不使用 most_popular 等推断边，也不注入 worry 噪声（α=0）。
    
    注意：这里的“共享”指边集合的生成机制一致，且不含个体噪声；每个节点仍只保留自身邻居。
    """
    rng = np.random.default_rng(seed)
    all_ids = set(class_data.student_ids)

    
    adj: Dict[int, Set[int]] = {sid: set() for sid in all_ids}
    for sid, student in class_data.students.items():
        for fid in student.friend_ids:
            if fid in class_data.students and fid != sid:
                adj[sid].add(fid)
                adj[fid].add(sid)

    # 2) 为每个学生生成一份“客观图视角”（无推断、无噪声）
    rashomon = RashomonSet(class_code=class_data.class_code)
    for sid, student in class_data.students.items():
        graph = SubjectiveGraph(
            owner_id=sid,
            class_code=class_data.class_code,
            nodes=set(all_ids),
            worry_level=student.worry_level,
            noise_alpha=0.0,
        )
        for nb in sorted(adj.get(sid, [])):
            graph.add_edge(nb, weight=1.0, edge_type="friend")
        rashomon.graphs[sid] = graph

    return rashomon


def build_all_rashomon_sets(
    split: DatasetSplit,
    worry_noise_scale: float = 1.0,
    seed: int = 42,
) -> Dict[str, RashomonSet]:
    """Build Rashomon sets for all classes.
    
    Args:
        split: Dataset split
        worry_noise_scale: Anxiety noise scaling factor
        seed: Random seed
    
    Returns:
        Dictionary mapping class_code -> RashomonSet
    """
    all_rashomon = {}
    
    for class_code, class_data in split.full_temporal.items():
        rashomon = build_rashomon_set(
            class_data=class_data,
            worry_noise_scale=worry_noise_scale,
            seed=seed + hash(class_code) % 10000,
        )
        all_rashomon[class_code] = rashomon
    
    return all_rashomon


def build_all_objective_friend_sets(
    split: DatasetSplit,
    seed: int = 42,
) -> Dict[str, RashomonSet]:
    """Build No-Subjective-Graph ablation version for all classes."""
    all_rashomon: Dict[str, RashomonSet] = {}
    for class_code, class_data in split.full_temporal.items():
        rashomon = build_objective_friend_rashomon_set(
            class_data=class_data,
            seed=seed + hash(class_code) % 10000,
        )
        all_rashomon[class_code] = rashomon
    return all_rashomon


def compute_graph_stats(rashomon: RashomonSet) -> Dict:
    """Compute statistics for Rashomon set."""
    degrees = [g.degree for g in rashomon.graphs.values()]
    alphas = [g.noise_alpha for g in rashomon.graphs.values()]
    
    edge_types = {"friend": 0, "inferred": 0, "noise": 0}
    for g in rashomon.graphs.values():
        for edge in g.edges.values():
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1
    
    return {
        "n_agents": rashomon.size,
        "mean_degree": np.mean(degrees),
        "std_degree": np.std(degrees),
        "min_degree": min(degrees),
        "max_degree": max(degrees),
        "mean_alpha": np.mean(alphas),
        "edge_type_counts": edge_types,
        "total_edges": sum(edge_types.values()),
    }

