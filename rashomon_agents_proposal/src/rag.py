"""RAG 检索引擎（本地实现）。

目标：在不引入外部向量库的前提下，实现“受限视野 + 可控噪声”的证据拼接，
并为 LLM 提示词提供结构化上下文。

实现要点：
- **Scope 硬约束**：self / friends / classmates / class_stats
- **主观图约束**：classmates 与 friends 默认受 `SubjectiveGraph` 可见边限制
- **焦虑噪声**：按 \(\alpha_i\) 对检索条目做删改/降置信度（当前实现仅做条目删改）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
import json
import re

import numpy as np

from .data_loader import StudentRecord, ClassData, DatasetSplit, get_all_students
from .subjective_graph import SubjectiveGraph, RashomonSet


class RetrievalScope(Enum):
    """检索范围枚举。"""
    SELF = "self"  # 仅自己的信息
    FRIENDS = "friends"  # 好友块内的信息
    CLASSMATES = "classmates"  # 同班同学（但受主观图约束）
    CLASS_STATS = "class_stats"  # 班级聚合统计


class RAGMode(Enum):
    """RAG 模式（用于消融实验）。"""
    SCOPED = "scoped"       # 默认：受主观图/范围约束
    NO_RAG = "no_rag"       # 消融：不提供目标相关检索（仅 self + class_stats）


@dataclass
class ScopeConstraint:
    """检索范围约束。"""
    allowed_scopes: Set[RetrievalScope] = field(default_factory=lambda: {
        RetrievalScope.SELF,
        RetrievalScope.FRIENDS,
        RetrievalScope.CLASS_STATS,
    })
    
    # 是否允许访问其他同学的详细信息（受主观图约束）
    allow_classmate_details: bool = False
    
    # 可访问的字段类型
    allowed_field_types: Set[str] = field(default_factory=lambda: {
        "personality",  # 性格特点
        "hobby",  # 爱好
        "friendship",  # 交友态度
        "exam_rank",  # 考试排名（非原始成绩）
        "class_stats",  # 班级统计
    })


@dataclass
class RetrievalItem:
    """单条检索结果。"""
    text: str  # 可直接拼接入提示词的证据片段
    source_cols: List[str]  # 来源列名
    confidence: float  # 置信度（用于后续噪声注入与权重调节）
    source_type: str = "direct"  # direct / inferred / aggregated
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "source_cols": self.source_cols,
            "confidence": self.confidence,
            "source_type": self.source_type,
        }


@dataclass
class RetrievalResult:
    """检索响应。"""
    agent_id: int
    query: str
    items: List[RetrievalItem] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "query": self.query,
            "items": [item.to_dict() for item in self.items],
        }
    
    def to_prompt_context(self) -> str:
        """将检索结果转换为提示词上下文。"""
        if not self.items:
            return "【检索结果】无相关信息。"
        
        lines = ["【检索结果】"]
        for i, item in enumerate(self.items, 1):
            lines.append(f"{i}. {item.text} (置信度: {item.confidence:.2f})")
        return "\n".join(lines)


def format_personality(student: StudentRecord) -> str:
    """格式化性格特点为自然语言。"""
    traits = []
    trait_map = {
        "extroverted_lively": "外向活泼",
        "introverted_quiet": "内向安静",
        "emotionally_stable": "情绪稳定",
        "emotionally_volatile": "情绪波动",
        "optimistic_positive": "乐观积极",
        "pessimistic": "悲观",
        "impulsive": "冲动",
        "thoughtful": "深思熟虑",
    }
    for key, chinese in trait_map.items():
        if student.personality.get(key, 0) == 1:
            traits.append(chinese)
    return "、".join(traits) if traits else "未知"


def format_exam_rank(student: StudentRecord, exam_idx: int = -1) -> str:
    """格式化考试排名信息。
    
    Args:
        student: 学生记录
        exam_idx: 考试索引（0-5），-1 表示最新
    """
    if exam_idx < 0:
        exam_idx = len(student.exam_class_ranks) - 1
    
    if exam_idx >= len(student.exam_class_ranks):
        return "排名未知"
    
    rank = student.exam_class_ranks[exam_idx]
    if np.isnan(rank):
        return "排名未知"
    
    return f"第{exam_idx+1}次考试班级排名第{int(rank)}名"


def format_student_summary(
    student: StudentRecord,
    include_personality: bool = True,
    include_rank: bool = True,
    exam_idx: int = -1,
) -> str:
    """生成学生的摘要描述（用于 RAG 检索）。"""
    parts = [f"学生{student.student_id}"]
    
    if include_personality:
        traits = format_personality(student)
        if traits != "未知":
            parts.append(f"性格特点: {traits}")
    
    if include_rank:
        rank_info = format_exam_rank(student, exam_idx)
        parts.append(rank_info)
    
    return "，".join(parts) + "。"


def compute_class_stats(class_data: ClassData, exam_idx: int = -1) -> Dict[str, Any]:
    """计算班级聚合统计。"""
    ranks = []
    for student in class_data.students.values():
        if exam_idx < 0:
            exam_idx = len(student.exam_class_ranks) - 1
        if exam_idx < len(student.exam_class_ranks):
            rank = student.exam_class_ranks[exam_idx]
            if not np.isnan(rank):
                ranks.append(rank)
    
    if not ranks:
        return {"mean_rank": None, "std_rank": None, "n_students": 0}
    
    return {
        "mean_rank": np.mean(ranks),
        "std_rank": np.std(ranks),
        "min_rank": min(ranks),
        "max_rank": max(ranks),
        "n_students": len(ranks),
    }


def inject_retrieval_noise(
    items: List[RetrievalItem],
    alpha: float,
    rng: np.random.Generator,
) -> List[RetrievalItem]:
    """对检索结果注入噪声。
    
    按 Proposal 规范，噪声可作用在：
    (i) 检索结果的删改
    (ii) 关系边的漏报/误报
    (iii) 事实到叙事的情绪化改写
    
    这里实现 (i)：以 α 概率删除部分检索结果
    """
    if alpha == 0 or not items:
        return items
    
    noisy_items = []
    for item in items:
        # 以 (1 - alpha * 0.5) 的概率保留
        if rng.random() > alpha * 0.5:
            # 降低置信度
            noisy_item = RetrievalItem(
                text=item.text,
                source_cols=item.source_cols,
                confidence=item.confidence * (1 - alpha * 0.3),
                source_type=item.source_type,
            )
            noisy_items.append(noisy_item)
    
    return noisy_items


class RAGEngine:
    """RAG 检索引擎。
    
    核心功能：
    - 为每个智能体提供受 scope 约束的检索接口
    - 结合主观图限制可见范围
    - 注入焦虑噪声
    """
    
    def __init__(
        self,
        split: DatasetSplit,
        rashomon_sets: Dict[str, RashomonSet],
        top_k: int = 8,
        seed: int = 42,
        mode: RAGMode = RAGMode.SCOPED,
    ):
        """
        Args:
            split: 数据集划分
            rashomon_sets: 所有班级的 Rashomon 集合
            top_k: 默认返回的最大结果数
            seed: 随机种子
        """
        self.split = split
        self.rashomon_sets = rashomon_sets
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        
        # 构建全局学生索引
        self.all_students = get_all_students(split)
        
        # 学生 ID 到班级的映射
        self.student_to_class: Dict[int, str] = {}
        for class_code, class_data in split.full_temporal.items():
            for sid in class_data.student_ids:
                self.student_to_class[sid] = class_code
    
    def retrieve(
        self,
        agent_id: int,
        query: str,
        scope: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        exam_idx: int = -1,
    ) -> RetrievalResult:
        """执行检索。
        
        Args:
            agent_id: 智能体 ID
            query: 查询字符串（用于语义匹配，当前版本用关键词）
            scope: 允许的检索范围列表
            top_k: 返回的最大结果数
            exam_idx: 考试时间点索引
        
        Returns:
            RetrievalResult
        """
        if top_k is None:
            top_k = self.top_k
        
        if scope is None:
            scope = ["self", "friends", "class_stats"]
        
        # 获取智能体信息
        if agent_id not in self.all_students:
            return RetrievalResult(agent_id=agent_id, query=query, items=[])
        
        agent = self.all_students[agent_id]
        class_code = self.student_to_class.get(agent_id)
        
        if class_code is None or class_code not in self.rashomon_sets:
            return RetrievalResult(agent_id=agent_id, query=query, items=[])
        
        # 获取智能体的主观图（若使用 scoped 模式）
        rashomon = self.rashomon_sets[class_code]
        subjective_graph = rashomon.graphs.get(agent_id)
        
        items: List[RetrievalItem] = []

        # 消融：No-RAG -> 仅返回最小上下文（self + class_stats）
        if self.mode == RAGMode.NO_RAG:
            scope = ["self", "class_stats"]
        
        # 1. Self scope: 自己的信息
        if "self" in scope:
            items.extend(self._retrieve_self(agent, exam_idx))
        
        # 2. Friends scope: 好友信息（默认受主观图约束；但如果主观图缺失则退化为显式好友块）
        if "friends" in scope:
            items.extend(self._retrieve_friends(
                agent, subjective_graph, exam_idx
            ))
        
        # 3. Classmates scope: 同班同学摘要（受主观图约束）
        if "classmates" in scope:
            items.extend(self._retrieve_classmates(
                agent, subjective_graph, exam_idx, query
            ))
        
        # 4. Class stats scope: 班级聚合统计
        if "class_stats" in scope:
            items.extend(self._retrieve_class_stats(agent, exam_idx))
        
        # 应用噪声（No-RAG 模式下不注入检索噪声，避免把“禁用检索”与“噪声删改”混在一起）
        if self.mode == RAGMode.SCOPED and subjective_graph is not None and subjective_graph.noise_alpha > 0:
            items = inject_retrieval_noise(
                items, subjective_graph.noise_alpha, self.rng
            )
        
        # 截断到 top_k
        items = items[:top_k]
        
        return RetrievalResult(agent_id=agent_id, query=query, items=items)
    
    def _retrieve_self(
        self, agent: StudentRecord, exam_idx: int
    ) -> List[RetrievalItem]:
        """检索自己的信息。"""
        items = []
        
        # 性格特点
        traits = format_personality(agent)
        if traits != "未知":
            items.append(RetrievalItem(
                text=f"我的性格特点: {traits}",
                source_cols=["personality_*"],
                confidence=1.0,
                source_type="direct",
            ))
        
        # 考试排名
        rank_info = format_exam_rank(agent, exam_idx)
        if "未知" not in rank_info:
            items.append(RetrievalItem(
                text=f"我的{rank_info}",
                source_cols=[f"e0{exam_idx+1}_total_score_class_rank_*"],
                confidence=1.0,
                source_type="direct",
            ))
        
        # 焦虑水平
        worry_map = {0: "从不担心", 1: "有时担心", 2: "经常担心", 3: "非常担心"}
        items.append(RetrievalItem(
            text=f"我对他人看法的担心程度: {worry_map.get(agent.worry_level, '未知')}",
            source_cols=["worry_about_others_*"],
            confidence=1.0,
            source_type="direct",
        ))
        
        return items
    
    def _retrieve_friends(
        self,
        agent: StudentRecord,
        graph: Optional[SubjectiveGraph],
        exam_idx: int,
    ) -> List[RetrievalItem]:
        """检索好友信息（受主观图约束）。"""
        items = []
        class_code = agent.class_code
        class_data = self.split.full_temporal.get(class_code)
        
        if class_data is None:
            return items

        # 优先：遍历主观图中的好友边（如果存在）
        if graph is not None:
            for target_id, edge in graph.edges.items():
                if edge.edge_type != "friend":
                    continue

                friend = class_data.students.get(target_id)
                if friend is None:
                    continue

                summary = format_student_summary(friend, exam_idx=exam_idx)
                items.append(RetrievalItem(
                    text=f"好友信息: {summary}",
                    source_cols=[f"friend_*_id={target_id}"],
                    confidence=edge.weight,
                    source_type="direct",
                ))
        else:
            # 退化：使用显式好友块
            for friend_id in agent.friend_ids:
                friend = class_data.students.get(friend_id)
                if friend is None:
                    continue
                summary = format_student_summary(friend, exam_idx=exam_idx)
                items.append(RetrievalItem(
                    text=f"好友信息: {summary}",
                    source_cols=[f"friend_*_id={friend_id}"],
                    confidence=1.0,
                    source_type="direct",
                ))
        
        return items
    
    def _retrieve_classmates(
        self,
        agent: StudentRecord,
        graph: Optional[SubjectiveGraph],
        exam_idx: int,
        query: str,
    ) -> List[RetrievalItem]:
        """检索同班同学信息（受主观图约束）。"""
        items = []
        class_code = agent.class_code
        class_data = self.split.full_temporal.get(class_code)
        
        if class_data is None:
            return items
        
        # 默认：只能看到主观图中的邻居；若主观图缺失则不返回同学信息
        if graph is None:
            return items

        visible_ids = set(graph.edges.keys())
        
        # 解析查询中的目标学生 ID
        target_ids = set()
        # 简单的 ID 提取：查找数字
        for match in re.finditer(r'\b(\d+)\b', query):
            try:
                tid = int(match.group(1))
                if tid in visible_ids:
                    target_ids.add(tid)
            except ValueError:
                pass
        
        # 如果查询中没有具体 ID，返回可见邻居的摘要
        if not target_ids:
            target_ids = visible_ids
        
        for target_id in target_ids:
            if target_id == agent.student_id:
                continue
            
            classmate = class_data.students.get(target_id)
            if classmate is None:
                continue
            
            edge = graph.edges.get(target_id)
            confidence = edge.weight if edge else 0.5
            
            summary = format_student_summary(classmate, exam_idx=exam_idx)
            items.append(RetrievalItem(
                text=f"同学信息: {summary}",
                source_cols=[f"student_id={target_id}"],
                confidence=confidence,
                source_type="inferred" if edge and edge.edge_type != "friend" else "direct",
            ))
        
        return items
    
    def _retrieve_class_stats(
        self, agent: StudentRecord, exam_idx: int
    ) -> List[RetrievalItem]:
        """检索班级聚合统计。"""
        items = []
        class_code = agent.class_code
        class_data = self.split.full_temporal.get(class_code)
        
        if class_data is None:
            return items
        
        stats = compute_class_stats(class_data, exam_idx)
        
        if stats["n_students"] > 0:
            items.append(RetrievalItem(
                text=f"班级第{exam_idx+1 if exam_idx >= 0 else 6}次考试统计: "
                     f"共{stats['n_students']}人参考, "
                     f"排名标准差{stats['std_rank']:.1f}",
                source_cols=["class_aggregate"],
                confidence=0.9,
                source_type="aggregated",
            ))
        
        return items
    
    def retrieve_for_evaluation(
        self,
        agent_id: int,
        target_ids: List[int],
        exam_idx: int = -1,
    ) -> RetrievalResult:
        """为评估任务检索信息。
        
        这是批处理评估的专用接口：智能体需要评估多个目标的能力。
        
        Args:
            agent_id: 评估者 ID
            target_ids: 被评估者 ID 列表
            exam_idx: 考试时间点索引
        
        Returns:
            RetrievalResult，包含评估所需的上下文
        """
        # 构造包含目标 ID 的查询
        query = f"评估学生 {','.join(map(str, target_ids))} 的学业能力"
        
        result = self.retrieve(
            agent_id=agent_id,
            query=query,
            scope=["self", "friends", "classmates", "class_stats"],
            exam_idx=exam_idx,
        )
        
        return result


#
# 说明：
# - 端到端运行/测试请使用 `python -m src.main --dry-run`；这里不再保留模块级脚手架入口。
