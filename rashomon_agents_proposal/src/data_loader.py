"""数据加载与样本筛选。

核心产物：
- `DatasetSplit.full_temporal`: 满足 6 次考试全覆盖（以班级 rank 列判定）且班级人数≥阈值的班级集合
- `DatasetSplit.social_observed_ids`: 至少填写 1 个好友 id 的学生集合
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

import numpy as np
import pandas as pd
import yaml


# 用于判断 6 次考试全覆盖的列（班级排名，覆盖率最高）
EXAM_CLASS_RANK_COLS = [
    "e01_total_score_class_rank_2024_id_filled",
    "e02_total_score_class_rank_2024_id_filled",
    "e03_total_score_class_rank_2024_id_filled",
    "e04_total_score_class_rank_2024_id_filled",
    "e05_total_score_class_rank_2024_id_filled",
    "e06_total_score_class_rank_2024_id_filled",
]

# 实际成绩列（覆盖率较低，但有值时可用）
EXAM_TOTAL_COLS = [
    "e01_total_score_2024_id_filled",
    "e02_total_score_2024_id_filled",
    "e03_total_score_2024_id_filled",
    "e04_total_score_2024_id_filled",
    "e05_total_score_2024_id_filled",
    "e06_total_score_2024_id_filled",
]

# 年级排名列（作为额外参考）
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
    """单个学生的结构化记录。"""
    student_id: int
    grade_code: str
    class_code: str
    gender: str
    
    # 性格与问卷
    personality: Dict[str, int] = field(default_factory=dict)
    worry_level: int = 0  # 0=never, 1=sometimes, 2=often, 3=very_concerned
    
    # 好友 id 列表（仅保留非空）
    friend_ids: List[int] = field(default_factory=list)
    
    # 6 次考试成绩（总分）
    exam_scores: List[float] = field(default_factory=list)
    # 6 次考试班级排名
    exam_class_ranks: List[float] = field(default_factory=list)
    
    # 原始行数据的引用（用于 RAG 检索）
    raw_row: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassData:
    """单个班级的数据。"""
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
    """数据集划分结果。"""
    full_temporal: Dict[str, ClassData]  # class_code -> ClassData
    social_observed_ids: Set[int]  # 至少有 1 个好友的学生 id
    
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
    """从 worry_about_others_* 独热列中提取焦虑等级 (0-3)。"""
    if row.get("worry_about_others_very_concerned", 0) == 1:
        return 3
    if row.get("worry_about_others_often", 0) == 1:
        return 2
    if row.get("worry_about_others_sometimes", 0) == 1:
        return 1
    return 0  # never_worry 或缺失


def extract_friend_ids(row: pd.Series) -> List[int]:
    """提取非空的好友 id 列表。"""
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
    """提取性格独热编码。"""
    return {col.replace("personality_", ""): int(row.get(col, 0) or 0) 
            for col in PERSONALITY_COLS}


def row_to_student(row: pd.Series) -> StudentRecord:
    """将 DataFrame 行转换为 StudentRecord。"""
    # 基础信息
    student_id = int(row["student_id"])
    grade_code = str(row.get("grade_code", ""))
    class_code = str(row.get("class_code", ""))
    gender = str(row.get("gender", "unknown"))
    
    # 问卷
    personality = extract_personality(row)
    worry_level = extract_worry_level(row)
    friend_ids = extract_friend_ids(row)
    
    # 成绩（班级排名作为主要指标，覆盖率高）
    exam_class_ranks = []
    for col in EXAM_CLASS_RANK_COLS:
        val = row.get(col)
        exam_class_ranks.append(float(val) if pd.notna(val) else np.nan)
    
    # 实际成绩（覆盖率较低，可能有缺失）
    exam_scores = []
    for col in EXAM_TOTAL_COLS:
        val = row.get(col)
        exam_scores.append(float(val) if pd.notna(val) else np.nan)
    
    # 原始行数据（转为 dict）
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
    """加载 CSV 并按条件筛选。
    
    Args:
        csv_path: 清洗后的 CSV 路径
        min_students_per_class: 班级最小人数阈值
        require_full_temporal: 是否要求 6 次考试全覆盖
    
    Returns:
        DatasetSplit 包含 full_temporal 班级和 social_observed 学生 id
    """
    df = pd.read_csv(csv_path)
    print(f"[DataLoader] 原始数据: {len(df)} 行, {df.shape[1]} 列")
    
    # Step 1: 筛选 6 次考试全覆盖的学生（使用班级排名列判断，覆盖率更高）
    if require_full_temporal:
        mask_temporal = df[EXAM_CLASS_RANK_COLS].notna().all(axis=1)
        df_temporal = df[mask_temporal].copy()
        print(f"[DataLoader] 6 次考试全覆盖（按班级排名）: {len(df_temporal)} 人")
    else:
        df_temporal = df.copy()
    
    # Step 2: 按班级分组，筛选满足人数阈值的班级
    class_counts = df_temporal.groupby("class_code").size()
    valid_classes = class_counts[class_counts >= min_students_per_class].index.tolist()
    df_filtered = df_temporal[df_temporal["class_code"].isin(valid_classes)].copy()
    print(f"[DataLoader] 满足班级人数阈值 (>={min_students_per_class}) 的班级数: {len(valid_classes)}")
    print(f"[DataLoader] 筛选后总人数: {len(df_filtered)}")
    
    # Step 3: 构建 ClassData 字典
    classes: Dict[str, ClassData] = {}
    social_observed_ids: Set[int] = set()
    
    for class_code in valid_classes:
        class_df = df_filtered[df_filtered["class_code"] == class_code]
        class_data = ClassData(class_code=class_code)
        
        for _, row in class_df.iterrows():
            student = row_to_student(row)
            class_data.students[student.student_id] = student
            
            # 检查是否有好友
            if len(student.friend_ids) > 0:
                social_observed_ids.add(student.student_id)
        
        classes[class_code] = class_data
    
    print(f"[DataLoader] Social-Observed 样本 (至少 1 个好友): {len(social_observed_ids)} 人")
    
    return DatasetSplit(
        full_temporal=classes,
        social_observed_ids=social_observed_ids,
    )


def load_config(config_path: str | Path) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_all_students(split: DatasetSplit) -> Dict[int, StudentRecord]:
    """获取所有学生的扁平字典。"""
    all_students = {}
    for class_data in split.full_temporal.values():
        all_students.update(class_data.students)
    return all_students


def get_class_student_ids(split: DatasetSplit, class_code: str) -> List[int]:
    """获取指定班级的学生 id 列表。"""
    if class_code not in split.full_temporal:
        return []
    return split.full_temporal[class_code].student_ids


def normalize_score_to_ability(
    scores: List[float], 
    class_scores: List[List[float]],
    method: str = "rank_percentile",
) -> List[float]:
    """将成绩映射到 [0, 1] 的能力刻度。
    
    Args:
        scores: 单个学生的 6 次成绩
        class_scores: 班级所有学生的成绩列表
        method: 归一化方法
            - "rank_percentile": 按班级排名百分位
            - "minmax": 按班级内 min-max 归一化
    
    Returns:
        归一化后的 6 个能力值
    """
    abilities = []
    n_exams = len(scores)
    
    for t in range(n_exams):
        score = scores[t]
        if np.isnan(score):
            abilities.append(0.5)  # 缺失值用中位数
            continue
        
        # 收集该考试的所有有效成绩
        valid_scores = [s[t] for s in class_scores if not np.isnan(s[t])]
        if len(valid_scores) == 0:
            abilities.append(0.5)
            continue
        
        if method == "rank_percentile":
            # 计算排名百分位（越高越好）
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


#
# 说明：
# - 不提供 `python -m src.data_loader` 形式的脚手架入口；请使用 `python -m src.main --dry-run`
#   来做端到端 sanity check（会同时覆盖数据加载、主观图、RAG、Agent、模拟器）。
