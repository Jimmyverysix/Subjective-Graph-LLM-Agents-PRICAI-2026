"""主观图构建（Rashomon set）。

每个学生 \(i\) 拥有自己的主观图 \(G_i=(V,E_i,\pi_i)\)，用于约束其可见邻域与信息可达性。

当前实现包含：
- **显式好友边**：来自 `friend_1_id` ~ `friend_6_id`
- **推断边（弱）**：来自 `most_popular_*_id`（作为“单向关注”的弱连边）
- **焦虑噪声**：按 \(\alpha_i=g(worry\_about\_others)\) 做漏边/误连边
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import random

import numpy as np

from .data_loader import StudentRecord, ClassData, DatasetSplit


@dataclass
class SubjectiveEdge:
    """主观图中的边。"""
    source: int  # 边的观察者（主观图的所有者）
    target: int  # 被观察的节点
    weight: float = 1.0  # 边的主观权重/可信度 π_i(e) ∈ [0, 1]
    edge_type: str = "friend"  # 边类型：friend, inferred, noise
    
    def __hash__(self):
        return hash((self.source, self.target))
    
    def __eq__(self, other):
        if not isinstance(other, SubjectiveEdge):
            return False
        return self.source == other.source and self.target == other.target


@dataclass
class SubjectiveGraph:
    """单个智能体的主观图 G_i = (V, E_i, π_i)。"""
    owner_id: int  # 主观图的所有者
    class_code: str
    
    # 班级内所有节点（不变）
    nodes: Set[int] = field(default_factory=set)
    
    # 主观边集合
    edges: Dict[int, SubjectiveEdge] = field(default_factory=dict)  # target_id -> edge
    
    # 焦虑水平（0-3）
    worry_level: int = 0
    
    # 噪声参数
    noise_alpha: float = 0.0  # α_i = g(worry_about_others)
    
    def get_neighbors(self) -> List[int]:
        """获取主观邻居列表。"""
        return list(self.edges.keys())
    
    def get_neighbor_weights(self) -> Dict[int, float]:
        """获取邻居及其权重。"""
        return {target: edge.weight for target, edge in self.edges.items()}
    
    def add_edge(self, target: int, weight: float = 1.0, edge_type: str = "friend"):
        """添加主观边。"""
        if target == self.owner_id:
            return  # 不添加自环
        if target not in self.nodes:
            return  # 目标不在班级内
        self.edges[target] = SubjectiveEdge(
            source=self.owner_id,
            target=target,
            weight=weight,
            edge_type=edge_type,
        )
    
    def remove_edge(self, target: int):
        """移除主观边。"""
        if target in self.edges:
            del self.edges[target]
    
    @property
    def degree(self) -> int:
        """主观度数。"""
        return len(self.edges)


@dataclass
class RashomonSet:
    """Rashomon 集合：班级内所有智能体的主观图集合。
    
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
    """将焦虑等级映射到噪声强度 α_i。
    
    按 Proposal 规范：0/0.33/0.66/1.0 的线性映射
    
    Args:
        worry_level: 0=never, 1=sometimes, 2=often, 3=very_concerned
        scale: 缩放因子（config 中的 worry_noise_scale）
    
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
    """对边集合应用焦虑噪声。
    
    按 Proposal 的公式：
    P((i,k) ∈ E_i) = (1 - α_i) · 1[k ∈ N_i^friend] + α_i · ε
    
    Args:
        edges: 原始边集合
        all_classmates: 班级内所有同学 id
        owner_id: 主观图所有者 id
        alpha: 噪声强度
        rng: 随机数生成器
        epsilon: 小概率误连边参数
    
    Returns:
        噪声注入后的边集合
    """
    if alpha == 0:
        return edges
    
    noisy_edges = {}
    
    # 1. 处理原始好友边：以 (1 - α) 的概率保留
    for target, edge in edges.items():
        if edge.edge_type == "friend":
            keep_prob = 1 - alpha
            if rng.random() < keep_prob:
                noisy_edges[target] = edge
            # 否则漏掉这条边（模拟焦虑导致的社交感知偏差）
        else:
            # 非好友边（如 inferred）保持不变
            noisy_edges[target] = edge
    
    # 2. 可能添加噪声边：以 α · ε 的概率对非邻居添加误连边
    non_neighbors = all_classmates - {owner_id} - set(noisy_edges.keys())
    noise_prob = alpha * epsilon
    
    for candidate in non_neighbors:
        if rng.random() < noise_prob:
            noisy_edges[candidate] = SubjectiveEdge(
                source=owner_id,
                target=candidate,
                weight=0.3,  # 噪声边权重较低
                edge_type="noise",
            )
    
    return noisy_edges


def build_subjective_graph(
    student: StudentRecord,
    class_data: ClassData,
    worry_noise_scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> SubjectiveGraph:
    """为单个学生构建主观图。
    
    Args:
        student: 学生记录
        class_data: 班级数据
        worry_noise_scale: 焦虑噪声缩放因子
        rng: 随机数生成器
    
    Returns:
        该学生的主观图
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # 初始化主观图
    graph = SubjectiveGraph(
        owner_id=student.student_id,
        class_code=student.class_code,
        nodes=set(class_data.student_ids),
        worry_level=student.worry_level,
        noise_alpha=worry_to_alpha(student.worry_level, worry_noise_scale),
    )
    
    # Step 1: 添加显式好友边
    for friend_id in student.friend_ids:
        if friend_id in class_data.students:
            # 好友在同班内
            graph.add_edge(friend_id, weight=1.0, edge_type="friend")
    
    # Step 2: 从 raw_row 中提取关系推断边（best_pair, most_popular）
    # best_pair: 认为关系好的两人（如果自己认识其中一人，可能对另一人有间接印象）
    # most_popular: 认为人缘好的人（可能有单向关注）
    raw = student.raw_row
    
    # 提取 most_popular_*_id
    for i in range(1, 4):
        col = f"most_popular_{i}_id"
        if col in raw and raw[col] is not None:
            try:
                pop_id = int(raw[col])
                if pop_id in class_data.students and pop_id != student.student_id:
                    if pop_id not in graph.edges:
                        # 添加推断边（权重较低）
                        graph.add_edge(pop_id, weight=0.5, edge_type="inferred")
            except (ValueError, TypeError):
                pass
    
    # Step 3: 应用焦虑噪声
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
    """为整个班级构建 Rashomon 集合。
    
    Args:
        class_data: 班级数据
        worry_noise_scale: 焦虑噪声缩放因子
        seed: 随机种子
    
    Returns:
        班级的 Rashomon 集合
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

    # 1) 用显式好友块构建对称邻接（undirected union）
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
    """为所有班级构建 Rashomon 集合。
    
    Args:
        split: 数据集划分
        worry_noise_scale: 焦虑噪声缩放因子
        seed: 随机种子
    
    Returns:
        class_code -> RashomonSet 的字典
    """
    all_rashomon = {}
    
    for class_code, class_data in split.full_temporal.items():
        rashomon = build_rashomon_set(
            class_data=class_data,
            worry_noise_scale=worry_noise_scale,
            seed=seed + hash(class_code) % 10000,  # 每个班级不同的种子
        )
        all_rashomon[class_code] = rashomon
    
    return all_rashomon


def build_all_objective_friend_sets(
    split: DatasetSplit,
    seed: int = 42,
) -> Dict[str, RashomonSet]:
    """为所有班级构建 No-Subjective-Graph 消融版本。"""
    all_rashomon: Dict[str, RashomonSet] = {}
    for class_code, class_data in split.full_temporal.items():
        rashomon = build_objective_friend_rashomon_set(
            class_data=class_data,
            seed=seed + hash(class_code) % 10000,
        )
        all_rashomon[class_code] = rashomon
    return all_rashomon


def compute_graph_stats(rashomon: RashomonSet) -> Dict:
    """计算 Rashomon 集合的统计信息。"""
    degrees = [g.degree for g in rashomon.graphs.values()]
    alphas = [g.noise_alpha for g in rashomon.graphs.values()]
    
    # 边类型统计
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

