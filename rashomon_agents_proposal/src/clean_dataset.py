from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd


NUMERIC_RE = re.compile(r"^\d+$")


def normalize_gender(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "unknown"
    s = str(x).strip().lower()
    if s in {"m", "male", "man", "boy", "1"}:
        return "male"
    if s in {"f", "female", "woman", "girl", "0"}:
        return "female"
    if s in {"unknown", "unk"}:
        return "unknown"
    # 数据中已基本为 male/female/unknown；其它情况统一 unknown
    return "unknown"


def numeric_only(series: pd.Series) -> pd.Series:
    # 仅保留纯数字字符串/数字，其余置空（NaN）
    def fix(v: object):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return pd.NA
        if isinstance(v, (int,)):
            return int(v)
        s = str(v).strip()
        if NUMERIC_RE.fullmatch(s):
            return int(s)
        return pd.NA

    return series.map(fix)


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 基础信息 → 问卷 → 成绩（尽量以列名规则自动推断）
    base_cols = [c for c in ["student_id", "grade_code", "class_code", "gender"] if c in df.columns]

    # 问卷：以你给的独热字段前缀为主，保持原始出现顺序
    survey_prefixes = (
        "personality_",
        "hobby_",
        "friendship_attitude_",
        "worry_about_others_",
        "feel_excluded_",
        "clique_presence_",
        "friend_",
        "best_pair_",
        "most_popular_",
    )
    survey_cols = [c for c in df.columns if c.startswith(survey_prefixes)]

    # 成绩：以 e0 开头
    score_cols = [c for c in df.columns if c.startswith("e0")]

    # 其它列（不丢弃，只放到问卷和成绩之间）
    used = set(base_cols) | set(survey_cols) | set(score_cols)
    other_cols = [c for c in df.columns if c not in used]

    ordered = base_cols + survey_cols + other_cols + score_cols
    # 去重保持顺序
    seen = set()
    final = []
    for c in ordered:
        if c in seen:
            continue
        if c in df.columns:
            final.append(c)
            seen.add(c)
    return df[final]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and reorder FINAL_full_plus_all_scores_cleaned.csv")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    df = pd.read_csv(inp)

    # gender normalization
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(normalize_gender)

    # numeric-only for *_id-like columns
    id_like = [c for c in df.columns if c.endswith("_id") or "_id_" in c]
    for c in id_like:
        df[c] = numeric_only(df[c])

    df = order_columns(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[clean_dataset] wrote: {out}  rows={len(df)} cols={df.shape[1]}")


if __name__ == "__main__":
    main()

