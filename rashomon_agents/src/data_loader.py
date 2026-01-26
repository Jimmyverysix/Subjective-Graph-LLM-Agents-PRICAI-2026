"""Data loading and sample filtering.

Key outputs:
- `DatasetSplit.full_temporal`: Classes with full 6-exam coverage (determined by class rank columns) and class size ≥ threshold
- `DatasetSplit.social_observed_ids`: Set of students with at least 1 friend ID filled
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

import numpy as np
import pandas as pd
import yaml


# Columns for determining full 6-exam coverage (class rank, highest coverage)
EXAM_CLASS_RANK_COLS = [
    "e01_total_score_class_rank_2024_id_filled",
    "e02_total_score_class_rank_2024_id_filled",
    "e03_total_score_class_rank_2024_id_filled",
    "e04_total_score_class_rank_2024_id_filled",
    "e05_total_score_class_rank_2024_id_filled",
    "e06_total_score_class_rank_2024_id_filled",
]

# Actual score columns (lower coverage, but usable when available)
EXAM_TOTAL_COLS = [
    "e01_total_score_2024_id_filled",
    "e02_total_score_2024_id_filled",
    "e03_total_score_2024_id_filled",
    "e04_total_score_2024_id_filled",
    "e05_total_score_2024_id_filled",
    "e06_total_score_2024_id_filled",
]

# Grade rank columns (as additional reference)
EXAM_GRADE_RANK_COLS = [
    "e01_total_score_grade_rank_2024_id_filled",
    "e02_total_score_grade_rank_2024_id_filled",
    "e03_total_score_grade_rank_2024_id_filled",
    "e04_total_score_grade_rank_2024_id_filled",
    "e05_total_score_grade_rank_2024_id_filled",
    "e06_total_score_grade_rank_2024_id_filled",
]

FRIEND_ID_COLS = [f"friend_{i}_id" for i in range(1, 7)]

PERSONALITY_COLS = [
    "personality_extroverted_lively",
    "personality_introverted_quiet",
    "personality_emotionally_stable",
    "personality_emotionally_volatile",
    "personality_optimistic_positive",
    "personality_pessimistic",
    "personality_impulsive",
    "personality_thoughtful",
]

WORRY_COLS = [
    "worry_about_others_never_worry",
    "worry_about_others_sometimes",
    "worry_about_others_often",
    "worry_about_others_very_concerned",
]


@dataclass
class StudentRecord:
    """Structured record for a single student."""
    student_id: int
    grade_code: str
    class_code: str
    gender: str
    
    personality: Dict[str, int] = field(default_factory=dict)
    worry_level: int = 0  # 0=never, 1=sometimes, 2=often, 3=very_concerned
    
    friend_ids: List[int] = field(default_factory=list)
    
    exam_scores: List[float] = field(default_factory=list)
    exam_class_ranks: List[float] = field(default_factory=list)
    
    raw_row: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassData:
    """Data for a single class."""
    class_code: str
    students: Dict[int, StudentRecord] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        return len(self.students)
    
    @property
    def student_ids(self) -> List[int]:
        return list(self.students.keys())


@dataclass
class DatasetSplit:
    """Dataset split result."""
    full_temporal: Dict[str, ClassData]
    social_observed_ids: Set[int]
    
    @property
    def n_full_temporal(self) -> int:
        return sum(c.size for c in self.full_temporal.values())
    
    @property
    def n_social_observed(self) -> int:
        return len(self.social_observed_ids)
    
    @property
    def n_classes(self) -> int:
        return len(self.full_temporal)


def extract_worry_level(row: pd.Series) -> int:
    """Extract worry level (0-3) from worry_about_others_* one-hot columns."""
    if row.get("worry_about_others_very_concerned", 0) == 1:
        return 3
    if row.get("worry_about_others_often", 0) == 1:
        return 2
    if row.get("worry_about_others_sometimes", 0) == 1:
        return 1
    return 0


def extract_friend_ids(row: pd.Series) -> List[int]:
    """Extract non-empty friend ID list."""
    ids = []
    for col in FRIEND_ID_COLS:
        val = row.get(col)
        if pd.notna(val):
            try:
                ids.append(int(val))
            except (ValueError, TypeError):
                pass
    return ids


def extract_personality(row: pd.Series) -> Dict[str, int]:
    """Extract personality one-hot encoding."""
    return {col.replace("personality_", ""): int(row.get(col, 0) or 0) 
            for col in PERSONALITY_COLS}


def row_to_student(row: pd.Series) -> StudentRecord:
    """Convert DataFrame row to StudentRecord."""
    student_id = int(row["student_id"])
    grade_code = str(row.get("grade_code", ""))
    class_code = str(row.get("class_code", ""))
    gender = str(row.get("gender", "unknown"))
    
    personality = extract_personality(row)
    worry_level = extract_worry_level(row)
    friend_ids = extract_friend_ids(row)
    
    exam_class_ranks = []
    for col in EXAM_CLASS_RANK_COLS:
        val = row.get(col)
        exam_class_ranks.append(float(val) if pd.notna(val) else np.nan)
    
    exam_scores = []
    for col in EXAM_TOTAL_COLS:
        val = row.get(col)
        exam_scores.append(float(val) if pd.notna(val) else np.nan)
    
    raw_row = row.to_dict()
    
    return StudentRecord(
        student_id=student_id,
        grade_code=grade_code,
        class_code=class_code,
        gender=gender,
        personality=personality,
        worry_level=worry_level,
        friend_ids=friend_ids,
        exam_scores=exam_scores,
        exam_class_ranks=exam_class_ranks,
        raw_row=raw_row,
    )


def load_and_filter(
    csv_path: str | Path,
    min_students_per_class: int = 30,
    require_full_temporal: bool = True,
) -> DatasetSplit:
    """Load CSV and filter by conditions.
    
    Args:
        csv_path: Path to cleaned CSV
        min_students_per_class: Minimum class size threshold
        require_full_temporal: Whether to require full 6-exam coverage
    
    Returns:
        DatasetSplit containing full_temporal classes and social_observed student IDs
    """
    df = pd.read_csv(csv_path)
    print(f"[DataLoader] Raw data: {len(df)} rows, {df.shape[1]} columns")
    
    if require_full_temporal:
        mask_temporal = df[EXAM_CLASS_RANK_COLS].notna().all(axis=1)
        df_temporal = df[mask_temporal].copy()
        print(f"[DataLoader] Full 6-exam coverage (by class rank): {len(df_temporal)} students")
    else:
        df_temporal = df.copy()
    
    class_counts = df_temporal.groupby("class_code").size()
    valid_classes = class_counts[class_counts >= min_students_per_class].index.tolist()
    df_filtered = df_temporal[df_temporal["class_code"].isin(valid_classes)].copy()
    print(f"[DataLoader] Classes meeting size threshold (>={min_students_per_class}): {len(valid_classes)}")
    print(f"[DataLoader] Total students after filtering: {len(df_filtered)}")
    
    classes: Dict[str, ClassData] = {}
    social_observed_ids: Set[int] = set()
    
    for class_code in valid_classes:
        class_df = df_filtered[df_filtered["class_code"] == class_code]
        class_data = ClassData(class_code=class_code)
        
        for _, row in class_df.iterrows():
            student = row_to_student(row)
            class_data.students[student.student_id] = student
            
            if len(student.friend_ids) > 0:
                social_observed_ids.add(student.student_id)
        
        classes[class_code] = class_data
    
    print(f"[DataLoader] Social-Observed samples (at least 1 friend): {len(social_observed_ids)} students")
    
    return DatasetSplit(
        full_temporal=classes,
        social_observed_ids=social_observed_ids,
    )


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_all_students(split: DatasetSplit) -> Dict[int, StudentRecord]:
    """Get flat dictionary of all students."""
    all_students = {}
    for class_data in split.full_temporal.values():
        all_students.update(class_data.students)
    return all_students


def get_class_student_ids(split: DatasetSplit, class_code: str) -> List[int]:
    """Get student ID list for specified class."""
    if class_code not in split.full_temporal:
        return []
    return split.full_temporal[class_code].student_ids


def normalize_score_to_ability(
    scores: List[float], 
    class_scores: List[List[float]],
    method: str = "rank_percentile",
) -> List[float]:
    """Map scores to [0, 1] ability scale.
    
    Args:
        scores: Single student's 6 exam scores
        class_scores: List of scores for all students in class
        method: Normalization method
            - "rank_percentile": By class rank percentile
            - "minmax": Min-max normalization within class
    
    Returns:
        Normalized 6 ability values
    """
    abilities = []
    n_exams = len(scores)
    
    for t in range(n_exams):
        score = scores[t]
        if np.isnan(score):
            abilities.append(0.5)
            continue
        
        valid_scores = [s[t] for s in class_scores if not np.isnan(s[t])]
        if len(valid_scores) == 0:
            abilities.append(0.5)
            continue
        
        if method == "rank_percentile":
            rank = sum(1 for s in valid_scores if s <= score)
            ability = rank / len(valid_scores)
        elif method == "minmax":
            min_s, max_s = min(valid_scores), max(valid_scores)
            if max_s > min_s:
                ability = (score - min_s) / (max_s - min_s)
            else:
                ability = 0.5
        else:
            ability = 0.5
        
        abilities.append(ability)
    
    return abilities


