"""RAG retrieval engine (local implementation).

Goal: Implement "limited-scope + controllable noise" evidence concatenation without external vector libraries,
and provide structured context for LLM prompts.

Key implementation points:
- **Scope hard constraints**: self / friends / classmates / class_stats
- **Subjective graph constraints**: classmates and friends are constrained by `SubjectiveGraph` visible edges by default
- **Anxiety noise**: Modify/ reduce confidence of retrieval items based on \(\alpha_i\) (current implementation only modifies items)
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
    """Retrieval scope enumeration."""
    SELF = "self"
    FRIENDS = "friends"
    CLASSMATES = "classmates"
    CLASS_STATS = "class_stats"


class RAGMode(Enum):
    """RAG mode (for ablation experiments)."""
    SCOPED = "scoped"
    NO_RAG = "no_rag"


@dataclass
class ScopeConstraint:
    """Retrieval scope constraints."""
    allowed_scopes: Set[RetrievalScope] = field(default_factory=lambda: {
        RetrievalScope.SELF,
        RetrievalScope.FRIENDS,
        RetrievalScope.CLASS_STATS,
    })
    
    allow_classmate_details: bool = False
    
    allowed_field_types: Set[str] = field(default_factory=lambda: {
        "personality",
        "hobby",
        "friendship",
        "exam_rank",
        "class_stats",
    })


@dataclass
class RetrievalItem:
    """Single retrieval result."""
    text: str
    source_cols: List[str]
    confidence: float
    source_type: str = "direct"
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "source_cols": self.source_cols,
            "confidence": self.confidence,
            "source_type": self.source_type,
        }


@dataclass
class RetrievalResult:
    """Retrieval response."""
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
        """Convert retrieval results to prompt context."""
        if not self.items:
            return "[Retrieval Results] No relevant information."
        
        lines = ["[Retrieval Results]"]
        for i, item in enumerate(self.items, 1):
            lines.append(f"{i}. {item.text} (Confidence: {item.confidence:.2f})")
        return "\n".join(lines)


def format_personality(student: StudentRecord) -> str:
    """Format personality traits to natural language."""
    traits = []
    trait_map = {
        "extroverted_lively": "Extroverted and lively",
        "introverted_quiet": "Introverted and quiet",
        "emotionally_stable": "Emotionally stable",
        "emotionally_volatile": "Emotionally volatile",
        "optimistic_positive": "Optimistic and positive",
        "pessimistic": "Pessimistic",
        "impulsive": "Impulsive",
        "thoughtful": "Thoughtful",
    }
    for key, english in trait_map.items():
        if student.personality.get(key, 0) == 1:
            traits.append(english)
    return ", ".join(traits) if traits else "Unknown"


def format_exam_rank(student: StudentRecord, exam_idx: int = -1) -> str:
    """Format exam rank information.
    
    Args:
        student: Student record
        exam_idx: Exam index (0-5), -1 means latest
    """
    if exam_idx < 0:
        exam_idx = len(student.exam_class_ranks) - 1
    
    if exam_idx >= len(student.exam_class_ranks):
        return "Rank unknown"
    
    rank = student.exam_class_ranks[exam_idx]
    if np.isnan(rank):
        return "Rank unknown"
    
    return f"Exam {exam_idx+1} class rank: {int(rank)}"


def format_student_summary(
    student: StudentRecord,
    include_personality: bool = True,
    include_rank: bool = True,
    exam_idx: int = -1,
) -> str:
    """Generate student summary description (for RAG retrieval)."""
    parts = [f"Student {student.student_id}"]
    
    if include_personality:
        traits = format_personality(student)
        if traits != "Unknown":
            parts.append(f"Personality: {traits}")
    
    if include_rank:
        rank_info = format_exam_rank(student, exam_idx)
        parts.append(rank_info)
    
    return ", ".join(parts) + "."


def compute_class_stats(class_data: ClassData, exam_idx: int = -1) -> Dict[str, Any]:
    """Compute class aggregate statistics."""
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
    """Inject noise into retrieval results.
    
    According to proposal specification, noise can act on:
    (i) Modification of retrieval results
    (ii) Missing/incorrect relationship edges
    (iii) Emotional rewriting from facts to narratives
    
    Here implements (i): Delete some retrieval results with probability α
    """
    if alpha == 0 or not items:
        return items
    
    noisy_items = []
    for item in items:
        if rng.random() > alpha * 0.5:
            noisy_item = RetrievalItem(
                text=item.text,
                source_cols=item.source_cols,
                confidence=item.confidence * (1 - alpha * 0.3),
                source_type=item.source_type,
            )
            noisy_items.append(noisy_item)
    
    return noisy_items


class RAGEngine:
    """RAG retrieval engine.
    
    Core functions:
    - Provide scope-constrained retrieval interface for each agent
    - Combine with subjective graph to limit visible scope
    - Inject anxiety noise
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
            split: Dataset split
            rashomon_sets: Rashomon sets for all classes
            top_k: Default maximum number of results to return
            seed: Random seed
        """
        self.split = split
        self.rashomon_sets = rashomon_sets
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        
        self.all_students = get_all_students(split)
        
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
        """Execute retrieval.
        
        Args:
            agent_id: Agent ID
            query: Query string (for semantic matching, current version uses keywords)
            scope: List of allowed retrieval scopes
            top_k: Maximum number of results to return
            exam_idx: Exam time point index
        
        Returns:
            RetrievalResult
        """
        if top_k is None:
            top_k = self.top_k
        
        if scope is None:
            scope = ["self", "friends", "class_stats"]
        
        
        if agent_id not in self.all_students:
            return RetrievalResult(agent_id=agent_id, query=query, items=[])
        
        agent = self.all_students[agent_id]
        class_code = self.student_to_class.get(agent_id)
        
        if class_code is None or class_code not in self.rashomon_sets:
            return RetrievalResult(agent_id=agent_id, query=query, items=[])
        
        
        rashomon = self.rashomon_sets[class_code]
        subjective_graph = rashomon.graphs.get(agent_id)
        
        items: List[RetrievalItem] = []

        
        if self.mode == RAGMode.NO_RAG:
            scope = ["self", "class_stats"]
        
        
        if "self" in scope:
            items.extend(self._retrieve_self(agent, exam_idx))
        
        
        if "friends" in scope:
            items.extend(self._retrieve_friends(
                agent, subjective_graph, exam_idx
            ))
        
        
        if "classmates" in scope:
            items.extend(self._retrieve_classmates(
                agent, subjective_graph, exam_idx, query
            ))
        
        
        if "class_stats" in scope:
            items.extend(self._retrieve_class_stats(agent, exam_idx))
        
        if self.mode == RAGMode.SCOPED and subjective_graph is not None and subjective_graph.noise_alpha > 0:
            items = inject_retrieval_noise(
                items, subjective_graph.noise_alpha, self.rng
            )
        
        
        items = items[:top_k]
        
        return RetrievalResult(agent_id=agent_id, query=query, items=items)
    
    def _retrieve_self(
        self, agent: StudentRecord, exam_idx: int
    ) -> List[RetrievalItem]:
        """Retrieve own information."""
        items = []
        
        traits = format_personality(agent)
        if traits != "Unknown":
            items.append(RetrievalItem(
                text=f"My personality traits: {traits}",
                source_cols=["personality_*"],
                confidence=1.0,
                source_type="direct",
            ))
        
        rank_info = format_exam_rank(agent, exam_idx)
        if "unknown" not in rank_info.lower():
            items.append(RetrievalItem(
                text=f"My {rank_info}",
                source_cols=[f"e0{exam_idx+1}_total_score_class_rank_*"],
                confidence=1.0,
                source_type="direct",
            ))
        
        worry_map = {0: "Never worry", 1: "Sometimes worry", 2: "Often worry", 3: "Very concerned"}
        items.append(RetrievalItem(
            text=f"My level of concern about others' opinions: {worry_map.get(agent.worry_level, 'Unknown')}",
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
        """Retrieve friend information (constrained by subjective graph)."""
        items = []
        class_code = agent.class_code
        class_data = self.split.full_temporal.get(class_code)
        
        if class_data is None:
            return items

        if graph is not None:
            for target_id, edge in graph.edges.items():
                if edge.edge_type != "friend":
                    continue

                friend = class_data.students.get(target_id)
                if friend is None:
                    continue

                summary = format_student_summary(friend, exam_idx=exam_idx)
                items.append(RetrievalItem(
                    text=f"Friend information: {summary}",
                    source_cols=[f"friend_*_id={target_id}"],
                    confidence=edge.weight,
                    source_type="direct",
                ))
        else:
            for friend_id in agent.friend_ids:
                friend = class_data.students.get(friend_id)
                if friend is None:
                    continue
                summary = format_student_summary(friend, exam_idx=exam_idx)
                items.append(RetrievalItem(
                    text=f"Friend information: {summary}",
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
        """Retrieve classmate information (constrained by subjective graph)."""
        items = []
        class_code = agent.class_code
        class_data = self.split.full_temporal.get(class_code)
        
        if class_data is None:
            return items
        
        if graph is None:
            return items

        visible_ids = set(graph.edges.keys())
        
        target_ids = set()
        for match in re.finditer(r'\b(\d+)\b', query):
            try:
                tid = int(match.group(1))
                if tid in visible_ids:
                    target_ids.add(tid)
            except ValueError:
                pass
        
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
                text=f"Classmate information: {summary}",
                source_cols=[f"student_id={target_id}"],
                confidence=confidence,
                source_type="inferred" if edge and edge.edge_type != "friend" else "direct",
            ))
        
        return items
    
    def _retrieve_class_stats(
        self, agent: StudentRecord, exam_idx: int
    ) -> List[RetrievalItem]:
        """Retrieve class aggregate statistics."""
        items = []
        class_code = agent.class_code
        class_data = self.split.full_temporal.get(class_code)
        
        if class_data is None:
            return items
        
        stats = compute_class_stats(class_data, exam_idx)
        
        if stats["n_students"] > 0:
            items.append(RetrievalItem(
                text=f"Class Exam {exam_idx+1 if exam_idx >= 0 else 6} Statistics: "
                     f"{stats['n_students']} students, "
                     f"rank std: {stats['std_rank']:.1f}",
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
        """Retrieve information for evaluation task.
        
        This is a dedicated interface for batch evaluation: agents need to evaluate the ability of multiple targets.
        
        Args:
            agent_id: Evaluator ID
            target_ids: List of target IDs to evaluate
            exam_idx: Exam time point index
        
        Returns:
            RetrievalResult containing context needed for evaluation
        """
        query = f"Evaluate academic ability of students {','.join(map(str, target_ids))}"
        
        result = self.retrieve(
            agent_id=agent_id,
            query=query,
            scope=["self", "friends", "classmates", "class_stats"],
            exam_idx=exam_idx,
        )
        
        return result


