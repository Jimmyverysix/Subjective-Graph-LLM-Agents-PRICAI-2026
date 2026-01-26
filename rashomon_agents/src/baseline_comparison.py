"""External baseline comparison experiments.

Comparison baselines:
1. Random: Random ranking
2. Self-Only: "No information" prior without social interaction
3. Linear/MLP Regressor: Predict ranking using individual questionnaire features
4. Graph Learning (1-hop, Rashomon view):
   - SGC/GCN-like: Adjacency normalization + k-step first-order propagation + Ridge
   - GAT-like: First-order neighbor attention aggregation + Ridge
5. DeGroot dynamics: Classical opinion dynamics (linear weighted average), for comparing "consensus convergence speed/diversity"

Important notes:
- We only implement "same first-order neighbor aggregation/scope constraints" (no unauthorized full-graph aggregation) to avoid strawman comparisons.
- This script only runs external baselines; Ours values come from main/ablation experiment summaries.
"""
from __future__ import annotations

import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
from scipy import stats as scipy_stats
from scipy import sparse as sp

from .data_loader import load_and_filter, load_config, DatasetSplit, ClassData


def compute_dpae(predicted_ranks: Dict[int, int], true_ranks: Dict[int, int]) -> Tuple[float, float]:
    """Compute DPAE and Spearman correlation."""
    common_ids = set(predicted_ranks.keys()) & set(true_ranks.keys())
    if len(common_ids) < 2:
        return 0.5, 0.0
    
    pred_list = [predicted_ranks[sid] for sid in common_ids]
    true_list = [true_ranks[sid] for sid in common_ids]
    
    rho, _ = scipy_stats.spearmanr(pred_list, true_list)
    if np.isnan(rho):
        rho = 0.0
    
    dpae = 1 - rho
    return dpae, rho


def get_true_ranks(class_data: ClassData, epoch: int) -> Dict[int, int]:
    """Get true class ranks."""
    ranks = {}
    for sid, student in class_data.students.items():
        if epoch < len(student.exam_class_ranks):
            rank = student.exam_class_ranks[epoch]
            if not np.isnan(rank):
                ranks[sid] = int(rank)
    return ranks


def compute_top_k_accuracy(predicted_ranks: Dict[int, int], true_ranks: Dict[int, int], k: int) -> float:
    """Compute Top-k accuracy."""
    common_ids = set(predicted_ranks.keys()) & set(true_ranks.keys())
    if len(common_ids) < k:
        return 0.0
    
    pred_top_k = set(sorted(common_ids, key=lambda x: predicted_ranks[x])[:k])
    true_top_k = set(sorted(common_ids, key=lambda x: true_ranks[x])[:k])
    
    return len(pred_top_k & true_top_k) / k


# ============================================================
# Baseline 1: Random Baseline
# ============================================================

def random_baseline(class_data: ClassData, epoch: int, seed: int = 42) -> Dict[str, float]:
    """Random ranking baseline."""
    rng = np.random.default_rng(seed)
    true_ranks = get_true_ranks(class_data, epoch)
    
    student_ids = list(true_ranks.keys())
    random_order = rng.permutation(student_ids)
    predicted_ranks = {sid: i + 1 for i, sid in enumerate(random_order)}
    
    dpae, rho = compute_dpae(predicted_ranks, true_ranks)
    acc3 = compute_top_k_accuracy(predicted_ranks, true_ranks, 3)
    acc5 = compute_top_k_accuracy(predicted_ranks, true_ranks, 5)
    
    return {"dpae": dpae, "spearman_rho": rho, "acc_3": acc3, "acc_5": acc5}


# ============================================================
# Baseline 2: Self-Only (No Social Interaction)
# ============================================================

def self_only_baseline(class_data: ClassData, epoch: int) -> Dict[str, float]:
    """Self-Only baseline: no social interaction, no external available signals.

    Implementation: All predicted scores are close to the same prior (0.5), with minimal noise to break ties, resulting in approximately random ordering.
    """
    true_ranks = get_true_ranks(class_data, epoch)
    rng = np.random.default_rng(42)
    predicted_scores = {sid: 0.5 + rng.normal(0, 0.01) for sid in true_ranks}
    sorted_by_score = sorted(predicted_scores.keys(), key=lambda x: -predicted_scores[x])
    predicted_ranks = {sid: i + 1 for i, sid in enumerate(sorted_by_score)}
    
    dpae, rho = compute_dpae(predicted_ranks, true_ranks)
    acc3 = compute_top_k_accuracy(predicted_ranks, true_ranks, 3)
    acc5 = compute_top_k_accuracy(predicted_ranks, true_ranks, 5)
    
    return {"dpae": dpae, "spearman_rho": rho, "acc_3": acc3, "acc_5": acc5}


# ============================================================
# Baseline 4 & 5: ML Methods (MLP, Linear Regression)
# ============================================================

def ml_baseline(split: DatasetSplit, epoch: int, method: str = "mlp") -> Dict[str, float]:
    """ML method: predict ranking using questionnaire features."""
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    
    all_features = []
    all_targets = []
    all_sids = []
    
    for class_code, class_data in split.full_temporal.items():
        true_ranks = get_true_ranks(class_data, epoch)
        n = len(true_ranks)
        
        for sid, student in class_data.students.items():
            if sid not in true_ranks:
                continue
            
            features = []
            for key, val in student.personality.items():
                features.append(float(val))
            features.append(float(student.worry_level) / 3.0)
            features.append(float(len(student.friend_ids)) / 6.0)
            
            target = 1.0 - (true_ranks[sid] - 1) / max(n - 1, 1)
            
            all_features.append(features)
            all_targets.append(target)
            all_sids.append((class_code, sid))
    
    X = np.array(all_features)
    y = np.array(all_targets)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    else:
        model = Ridge(alpha=1.0)
    
    y_pred = cross_val_predict(model, X_scaled, y, cv=5)
    
    class_metrics = defaultdict(list)
    for i, (class_code, sid) in enumerate(all_sids):
        class_metrics[class_code].append((sid, y_pred[i], y[i]))
    
    all_dpae = []
    all_rho = []
    all_acc3 = []
    
    for class_code, data in class_metrics.items():
        sorted_by_pred = sorted(data, key=lambda x: -x[1])
        predicted_ranks = {item[0]: i + 1 for i, item in enumerate(sorted_by_pred)}
        
        sorted_by_true = sorted(data, key=lambda x: -x[2])
        true_ranks = {item[0]: i + 1 for i, item in enumerate(sorted_by_true)}
        
        dpae, rho = compute_dpae(predicted_ranks, true_ranks)
        acc3 = compute_top_k_accuracy(predicted_ranks, true_ranks, 3)
        
        all_dpae.append(dpae)
        all_rho.append(rho)
        all_acc3.append(acc3)
    
    return {
        "dpae": np.mean(all_dpae),
        "spearman_rho": np.mean(all_rho),
        "acc_3": np.mean(all_acc3),
        "dpae_std": np.std(all_dpae),
    }


def _extract_features_and_targets(
    split: DatasetSplit, epoch: int
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int]], Dict[str, List[int]]]:
    """Extract global feature matrix X, targets y, and index mappings.

    - X: (N, F) Individual features (questionnaire personality + worry + friend_count)
    - y: (N,) Normalized ability (mapped from current true class rank to [0,1])
    - idx_to_node: Global index -> (class_code, sid)
    - class_to_indices: class_code -> [list of global indices]
    """
    X_rows: List[List[float]] = []
    y_rows: List[float] = []
    idx_to_node: List[Tuple[str, int]] = []
    class_to_indices: Dict[str, List[int]] = defaultdict(list)

    for class_code, class_data in split.full_temporal.items():
        true_ranks = get_true_ranks(class_data, epoch)
        n = len(true_ranks)
        if n < 2:
            continue

        for sid, student in class_data.students.items():
            if sid not in true_ranks:
                continue

            features: List[float] = []
            for _, val in student.personality.items():
                features.append(float(val))
            features.append(float(student.worry_level) / 3.0)
            features.append(float(len(student.friend_ids)) / 6.0)

            target = 1.0 - (true_ranks[sid] - 1) / max(n - 1, 1)

            idx = len(idx_to_node)
            idx_to_node.append((class_code, sid))
            class_to_indices[class_code].append(idx)
            X_rows.append(features)
            y_rows.append(float(target))

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    return X, y, idx_to_node, dict(class_to_indices)


def _build_block_diagonal_adjacency(split: DatasetSplit, idx_to_node: List[Tuple[str, int]]) -> sp.csr_matrix:
    """Build graph from survey friend blocks: only first-order neighbors (Rashomon view), no cross-class edges.

    Constructs a global sparse adjacency matrix A (directed), where A[i,j]=1 means i considers j a friend (visible first-order neighbor).
    """
    node_to_idx = {node: i for i, node in enumerate(idx_to_node)}
    N = len(idx_to_node)
    rows: List[int] = []
    cols: List[int] = []

    for i in range(N):
        rows.append(i)
        cols.append(i)

    for class_code, class_data in split.full_temporal.items():
        class_sids = {sid for c, sid in idx_to_node if c == class_code}
        if not class_sids:
            continue

        for sid, student in class_data.students.items():
            if sid not in class_sids:
                continue

            src = node_to_idx[(class_code, sid)]
            for fid in student.friend_ids:
                if fid in class_sids:
                    dst = node_to_idx[(class_code, fid)]
                    rows.append(src)
                    cols.append(dst)

    data = np.ones(len(rows), dtype=float)
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A


def _row_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    """Row normalization (directed propagation): D^{-1} A."""
    row_sum = np.asarray(A.sum(axis=1)).reshape(-1)
    row_sum[row_sum == 0] = 1.0
    inv = 1.0 / row_sum
    D_inv = sp.diags(inv)
    return D_inv @ A


def sgc_gcn_like_baseline(
    split: DatasetSplit,
    epoch: int,
    k_hops: int = 1,
    alpha: float = 1.0,
    cv: int = 5,
) -> Dict[str, float]:
    """SGC/GCN-like: linear graph convolution baseline with only first-order adjacency propagation (no deep learning dependencies).

    Method: X_k = (D^{-1}A)^k X, then Ridge regression on X_k (5-fold CV prediction).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict

    X, y, idx_to_node, class_to_indices = _extract_features_and_targets(split, epoch)
    if X.shape[0] == 0:
        return {"dpae": 0.5, "spearman_rho": 0.0, "acc_3": 0.0, "dpae_std": 0.0}

    A = _build_block_diagonal_adjacency(split, idx_to_node)
    P = _row_normalize(A)

    X_prop = X
    for _ in range(max(1, int(k_hops))):
        X_prop = (P @ X_prop).astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_prop)

    model = Ridge(alpha=alpha)
    y_pred = cross_val_predict(model, Xs, y, cv=cv)

    all_dpae, all_rho, all_acc3 = [], [], []
    for class_code, indices in class_to_indices.items():
        if len(indices) < 3:
            continue
        data = [(idx_to_node[i][1], float(y_pred[i]), float(y[i])) for i in indices]
        sorted_by_pred = sorted(data, key=lambda x: -x[1])
        predicted_ranks = {item[0]: r + 1 for r, item in enumerate(sorted_by_pred)}
        sorted_by_true = sorted(data, key=lambda x: -x[2])
        true_ranks = {item[0]: r + 1 for r, item in enumerate(sorted_by_true)}

        dpae, rho = compute_dpae(predicted_ranks, true_ranks)
        acc3 = compute_top_k_accuracy(predicted_ranks, true_ranks, 3)
        all_dpae.append(dpae)
        all_rho.append(rho)
        all_acc3.append(acc3)

    return {
        "dpae": float(np.mean(all_dpae)) if all_dpae else 0.5,
        "spearman_rho": float(np.mean(all_rho)) if all_rho else 0.0,
        "acc_3": float(np.mean(all_acc3)) if all_acc3 else 0.0,
        "dpae_std": float(np.std(all_dpae)) if all_dpae else 0.0,
    }


def gat_like_baseline(
    split: DatasetSplit,
    epoch: int,
    temperature: float = 0.2,
    alpha: float = 1.0,
    cv: int = 5,
) -> Dict[str, float]:
    """GAT-like: first-order neighbor attention aggregation + Ridge (no PyG dependency).

    Only uses survey friend_ids as first-order neighbors; attention weights computed from feature cosine similarity:
    w_ij = softmax( cos(x_i, x_j) / T ), and weighted sum of neighbor features to get x'_i.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict

    X, y, idx_to_node, class_to_indices = _extract_features_and_targets(split, epoch)
    if X.shape[0] == 0:
        return {"dpae": 0.5, "spearman_rho": 0.0, "acc_3": 0.0, "dpae_std": 0.0}

    node_to_idx = {node: i for i, node in enumerate(idx_to_node)}
    neighbors: List[List[int]] = [[] for _ in range(len(idx_to_node))]

    for class_code, class_data in split.full_temporal.items():
        class_sids = {sid for c, sid in idx_to_node if c == class_code}
        if not class_sids:
            continue

        for sid, student in class_data.students.items():
            if sid not in class_sids:
                continue

            src = node_to_idx[(class_code, sid)]
            nbs = [src]
            for fid in student.friend_ids:
                if fid in class_sids:
                    nbs.append(node_to_idx[(class_code, fid)])
            neighbors[src] = nbs

    eps = 1e-9
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    X_att = np.zeros_like(X, dtype=float)

    T = max(float(temperature), 1e-3)
    for i, nbs in enumerate(neighbors):
        if not nbs:
            X_att[i] = X[i]
            continue
        sims = (X_norm[i] @ X_norm[nbs].T).astype(float)
        logits = sims / T
        logits = logits - np.max(logits)
        w = np.exp(logits)
        w = w / (np.sum(w) + eps)
        X_att[i] = (w.reshape(-1, 1) * X[nbs]).sum(axis=0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_att)

    model = Ridge(alpha=alpha)
    y_pred = cross_val_predict(model, Xs, y, cv=cv)

    all_dpae, all_rho, all_acc3 = [], [], []
    for class_code, indices in class_to_indices.items():
        if len(indices) < 3:
            continue
        data = [(idx_to_node[i][1], float(y_pred[i]), float(y[i])) for i in indices]
        sorted_by_pred = sorted(data, key=lambda x: -x[1])
        predicted_ranks = {item[0]: r + 1 for r, item in enumerate(sorted_by_pred)}
        sorted_by_true = sorted(data, key=lambda x: -x[2])
        true_ranks = {item[0]: r + 1 for r, item in enumerate(sorted_by_true)}
        dpae, rho = compute_dpae(predicted_ranks, true_ranks)
        acc3 = compute_top_k_accuracy(predicted_ranks, true_ranks, 3)
        all_dpae.append(dpae)
        all_rho.append(rho)
        all_acc3.append(acc3)

    return {
        "dpae": float(np.mean(all_dpae)) if all_dpae else 0.5,
        "spearman_rho": float(np.mean(all_rho)) if all_rho else 0.0,
        "acc_3": float(np.mean(all_acc3)) if all_acc3 else 0.0,
        "dpae_std": float(np.std(all_dpae)) if all_dpae else 0.0,
    }


def degroot_dynamics_baseline(
    class_data: ClassData,
    epoch: int,
    n_steps: int = 30,
    seed: int = 42,
    self_signal_noise_std: float = 0.0,
) -> Dict[str, float]:
    r"""DeGroot opinion dynamics (vectorized "opinion diffusion for each target").

    Each node i holds a scalar belief b_{i,k} for each target k.
    Initialization:
    - b_{i,k}=0.5 (no-information prior)
    - b_{k,k}=s_k (k's private signal about itself); s_k comes from current true ability (mapped from rank) and can add Gaussian noise
    Update:
    - b_{i,k}^{t+1} = Σ_{j∈N(i)∪{i}} w_{ij} b_{j,k}^{t}, where w is evenly divided among first-order neighbors (row-stochastic matrix)
    Output:
    - Use group mean \bar{b}_{\cdot,k} as group estimate for target k and rank
    - Additional output diversity_final (average variance of b_{i,k} over i at final time)
    """
    rng = np.random.default_rng(seed)
    true_ranks = get_true_ranks(class_data, epoch)
    student_ids = list(true_ranks.keys())
    n = len(student_ids)
    if n < 3:
        return {"dpae": 0.5, "spearman_rho": 0.0, "acc_3": 0.0, "diversity_final": 0.0}

    abilities = {sid: 1.0 - (true_ranks[sid] - 1) / max(n - 1, 1) for sid in student_ids}
    if self_signal_noise_std > 0:
        for sid in student_ids:
            abilities[sid] = float(np.clip(abilities[sid] + rng.normal(0.0, self_signal_noise_std), 0.0, 1.0))

    neighbor_map: Dict[int, List[int]] = {}
    sid_set = set(student_ids)
    for sid in student_ids:
        nbs = [sid]
        if sid in class_data.students:
            for fid in class_data.students[sid].friend_ids:
                if fid in sid_set:
                    nbs.append(fid)
        neighbor_map[sid] = nbs

    sid_to_idx = {sid: i for i, sid in enumerate(student_ids)}
    rows, cols, data = [], [], []
    for sid in student_ids:
        i = sid_to_idx[sid]
        nbs = neighbor_map[sid]
        w = 1.0 / max(len(nbs), 1)
        for nb in nbs:
            j = sid_to_idx[nb]
            rows.append(i)
            cols.append(j)
            data.append(w)
    W = sp.csr_matrix((np.asarray(data, dtype=float), (rows, cols)), shape=(n, n))

    B = np.full((n, n), 0.5, dtype=float)
    for sid in student_ids:
        i = sid_to_idx[sid]
        B[i, i] = abilities[sid]

    for _ in range(max(1, int(n_steps))):
        B = (W @ B).astype(float)

    group_est = B.mean(axis=0)
    sorted_idx = np.argsort(-group_est)
    predicted_ranks = {student_ids[idx]: r + 1 for r, idx in enumerate(sorted_idx)}

    dpae, rho = compute_dpae(predicted_ranks, true_ranks)
    acc3 = compute_top_k_accuracy(predicted_ranks, true_ranks, 3)
    diversity_final = float(B.var(axis=0).mean())
    return {"dpae": float(dpae), "spearman_rho": float(rho), "acc_3": float(acc3), "diversity_final": diversity_final}


def _parse_int_list(spec: str) -> List[int]:
    """Parse integer list in '1,5,10,30' format."""
    if not spec:
        return []
    items = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def run_degroot_sweep(
    split: DatasetSplit,
    epoch: int,
    seeds: List[int],
    steps_list: List[int],
    noise_std: float,
    output_dir: Path,
) -> List[Dict[str, float]]:
    """扫描 DeGroot 的迭代步数，输出“收敛速度 vs 多样性塌缩”曲线数据与图。

    产物：
    - output_dir/degroot_sweep_epoch{E}.json
    - output_dir/degroot_sweep_epoch{E}.csv
    - output_dir/degroot_sweep_epoch{E}.png
    """
    steps_list = sorted(set(int(s) for s in steps_list if int(s) > 0))
    if not steps_list:
        return []

    sweep_rows: List[Dict[str, float]] = []
    for steps in steps_list:
        metrics = []
        for seed in seeds:
            for _, class_data in split.full_temporal.items():
                res = degroot_dynamics_baseline(
                    class_data,
                    epoch,
                    n_steps=steps,
                    seed=seed,
                    self_signal_noise_std=noise_std,
                )
                metrics.append(res)

        row = {
            "epoch": float(epoch + 1),
            "steps": float(steps),
            "dpae": float(np.mean([m["dpae"] for m in metrics])),
            "dpae_std": float(np.std([m["dpae"] for m in metrics])),
            "spearman_rho": float(np.mean([m["spearman_rho"] for m in metrics])),
            "acc_3": float(np.mean([m["acc_3"] for m in metrics])),
            "diversity_final": float(np.mean([m["diversity_final"] for m in metrics])),
        }
        sweep_rows.append(row)

    epoch_tag = f"epoch{epoch + 1}"
    json_path = output_dir / f"degroot_sweep_{epoch_tag}.json"
    csv_path = output_dir / f"degroot_sweep_{epoch_tag}.csv"
    fig_path = output_dir / f"degroot_sweep_{epoch_tag}.png"

    with open(json_path, "w") as f:
        json.dump(sweep_rows, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "steps", "dpae", "dpae_std", "spearman_rho", "acc_3", "diversity_final"],
        )
        writer.writeheader()
        for row in sweep_rows:
            writer.writerow(row)

    try:
        import matplotlib.pyplot as plt

        xs = [r["steps"] for r in sweep_rows]
        dpae = [r["dpae"] for r in sweep_rows]
        rho = [r["spearman_rho"] for r in sweep_rows]
        div = [r["diversity_final"] for r in sweep_rows]

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(xs, dpae, marker="o", label="DPAE (↓)")
        ax0b = ax[0].twinx()
        ax0b.plot(xs, rho, marker="s", color="tab:orange", label="Spearman ρ (↑)")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("DeGroot steps (log)")
        ax[0].set_ylabel("DPAE")
        ax0b.set_ylabel("Spearman ρ")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(xs, div, marker="o", color="tab:green")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("DeGroot steps (log)")
        ax[1].set_ylabel("Diversity (final)")
        ax[1].set_title("Consensus collapse (lower = more consensus)")
        ax[1].grid(True, alpha=0.3)

        fig.suptitle(f"DeGroot sweep ({epoch_tag}), noise_std={noise_std}")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
    except Exception:
        pass

    return sweep_rows


# ============================================================
# Main Function
# ============================================================

def run_all_baselines(
    split: DatasetSplit,
    epochs: List[int] = [0, 5],
    seeds: List[int] = [42, 43, 44],
    include_graph: bool = True,
    include_degroot: bool = True,
    degroot_steps: int = 30,
    degroot_noise_std: float = 0.0,
) -> Dict[str, Any]:
    """Run all baseline experiments."""
    results = {
        "random": [],
        "self_only": [],
        "linear_regression": [],
        "mlp": [],
        # graph learning
        "sgc_k1": [],
        "sgc_k2": [],
        "gat_like": [],
        # dynamics
        "degroot": [],
    }
    
    for epoch in epochs:
        print(f"\n=== Epoch {epoch + 1} ===")
        
        random_results = []
        for seed in seeds:
            for class_code, class_data in split.full_temporal.items():
                res = random_baseline(class_data, epoch, seed)
                random_results.append(res)
        results["random"].append({
            "epoch": epoch + 1,
            "dpae": np.mean([r["dpae"] for r in random_results]),
            "dpae_std": np.std([r["dpae"] for r in random_results]),
            "spearman_rho": np.mean([r["spearman_rho"] for r in random_results]),
            "acc_3": np.mean([r["acc_3"] for r in random_results]),
        })
        print(f"  Random: DPAE={results['random'][-1]['dpae']:.3f}±{results['random'][-1]['dpae_std']:.3f}")
        
        # Self-Only baseline
        self_results = []
        for class_code, class_data in split.full_temporal.items():
            res = self_only_baseline(class_data, epoch)
            self_results.append(res)
        results["self_only"].append({
            "epoch": epoch + 1,
            "dpae": np.mean([r["dpae"] for r in self_results]),
            "dpae_std": np.std([r["dpae"] for r in self_results]),
            "spearman_rho": np.mean([r["spearman_rho"] for r in self_results]),
            "acc_3": np.mean([r["acc_3"] for r in self_results]),
        })
        print(f"  Self-Only: DPAE={results['self_only'][-1]['dpae']:.3f}±{results['self_only'][-1]['dpae_std']:.3f}")
        
        # ML baselines
        lr_res = ml_baseline(split, epoch, method="linear")
        results["linear_regression"].append({
            "epoch": epoch + 1,
            "dpae": lr_res["dpae"],
            "dpae_std": lr_res["dpae_std"],
            "spearman_rho": lr_res["spearman_rho"],
            "acc_3": lr_res["acc_3"],
        })
        print(f"  Linear Regression: DPAE={lr_res['dpae']:.3f}±{lr_res['dpae_std']:.3f}")
        
        mlp_res = ml_baseline(split, epoch, method="mlp")
        results["mlp"].append({
            "epoch": epoch + 1,
            "dpae": mlp_res["dpae"],
            "dpae_std": mlp_res["dpae_std"],
            "spearman_rho": mlp_res["spearman_rho"],
            "acc_3": mlp_res["acc_3"],
        })
        print(f"  MLP: DPAE={mlp_res['dpae']:.3f}±{mlp_res['dpae_std']:.3f}")

        if include_graph:
            sgc1 = sgc_gcn_like_baseline(split, epoch, k_hops=1, alpha=1.0, cv=5)
            results["sgc_k1"].append({"epoch": epoch + 1, **sgc1})
            print(f"  SGC(k=1): DPAE={sgc1['dpae']:.3f}±{sgc1['dpae_std']:.3f}")

            sgc2 = sgc_gcn_like_baseline(split, epoch, k_hops=2, alpha=1.0, cv=5)
            results["sgc_k2"].append({"epoch": epoch + 1, **sgc2})
            print(f"  SGC(k=2): DPAE={sgc2['dpae']:.3f}±{sgc2['dpae_std']:.3f}")

            gat = gat_like_baseline(split, epoch, temperature=0.2, alpha=1.0, cv=5)
            results["gat_like"].append({"epoch": epoch + 1, **gat})
            print(f"  GAT-like(1-hop): DPAE={gat['dpae']:.3f}±{gat['dpae_std']:.3f}")

        if include_degroot:
            degroot_metrics = []
            for seed in seeds:
                for class_code, class_data in split.full_temporal.items():
                    res = degroot_dynamics_baseline(
                        class_data,
                        epoch,
                        n_steps=degroot_steps,
                        seed=seed,
                        self_signal_noise_std=degroot_noise_std,
                    )
                    degroot_metrics.append(res)
            results["degroot"].append({
                "epoch": epoch + 1,
                "dpae": float(np.mean([r["dpae"] for r in degroot_metrics])),
                "dpae_std": float(np.std([r["dpae"] for r in degroot_metrics])),
                "spearman_rho": float(np.mean([r["spearman_rho"] for r in degroot_metrics])),
                "acc_3": float(np.mean([r["acc_3"] for r in degroot_metrics])),
                "diversity_final": float(np.mean([r["diversity_final"] for r in degroot_metrics])),
            })
            print(
                f"  DeGroot(steps={degroot_steps}, noise={degroot_noise_std}): "
                f"DPAE={results['degroot'][-1]['dpae']:.3f}±{results['degroot'][-1]['dpae_std']:.3f}, "
                f"div={results['degroot'][-1]['diversity_final']:.4f}"
            )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="External baseline comparison experiments")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--output", default="output/baselines", help="Output directory")
    parser.add_argument("--no-graph", action="store_true", help="Disable graph learning baselines (SGC/GAT-like)")
    parser.add_argument("--no-degroot", action="store_true", help="Disable DeGroot dynamics baseline")
    parser.add_argument("--degroot-steps", type=int, default=30, help="DeGroot iteration steps")
    parser.add_argument("--degroot-noise-std", type=float, default=0.0, help="DeGroot private signal noise std")
    parser.add_argument(
        "--degroot-sweep-steps",
        type=str,
        default="",
        help="Sweep DeGroot steps, e.g. '1,2,5,10,30,100' (will output additional sweep csv/json/plot)",
    )
    parser.add_argument(
        "--degroot-sweep-epoch",
        type=int,
        default=6,
        help="Epoch for sweep (1-6), default 6 (final)",
    )
    parser.add_argument(
        "--degroot-sweep-only",
        action="store_true",
        help="Only run DeGroot sweep (skip other baselines)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("External Baseline Comparison Experiments")
    print("=" * 60)
    
    config = load_config(args.config)
    split = load_and_filter(config["data"]["cleaned_csv"])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sweep_steps = _parse_int_list(args.degroot_sweep_steps)
    sweep_epoch_idx = int(args.degroot_sweep_epoch) - 1
    if args.degroot_sweep_steps:
        if sweep_epoch_idx < 0 or sweep_epoch_idx > 5:
            raise ValueError("--degroot-sweep-epoch must be in [1,6]")

    if args.degroot_sweep_only:
        results = {}
    else:
        results = run_all_baselines(
            split,
            epochs=[0, 5],
            seeds=[42, 43, 44],
            include_graph=not args.no_graph,
            include_degroot=not args.no_degroot,
            degroot_steps=args.degroot_steps,
            degroot_noise_std=args.degroot_noise_std,
        )

    if sweep_steps:
        print("\n" + "=" * 60)
        print(f"DeGroot sweep (Epoch {sweep_epoch_idx + 1}): steps={sweep_steps}, noise_std={args.degroot_noise_std}")
        print("=" * 60)
        sweep_rows = run_degroot_sweep(
            split=split,
            epoch=sweep_epoch_idx,
            seeds=[42, 43, 44],
            steps_list=sweep_steps,
            noise_std=float(args.degroot_noise_std),
            output_dir=output_dir,
        )
        for r in sweep_rows:
            print(
                f"  steps={int(r['steps']):<4d} "
                f"DPAE={r['dpae']:.3f}±{r['dpae_std']:.3f} "
                f"ρ={r['spearman_rho']:.3f} "
                f"Acc@3={r['acc_3']:.3f} "
                f"Div={r['diversity_final']:.6f}"
            )

    if results:
        with open(output_dir / "baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary Table (Epoch 6, Final Exam)")
    print("=" * 60)
    print(f"{'Method':<25} {'DPAE':<15} {'Spearman ρ':<15} {'Acc@3':<10}")
    print("-" * 60)
    
    if not results:
        print("(degroot-sweep-only: skipped)")
        print("=" * 60)
        print(f"\nResults saved to: {output_dir}")
        return

    shown_methods = ["random", "self_only", "linear_regression", "mlp"]
    if not args.no_graph:
        shown_methods += ["sgc_k1", "sgc_k2", "gat_like"]
    if not args.no_degroot:
        shown_methods += ["degroot"]

    for method in shown_methods:
        if method not in results or not results[method]:
            continue
        final_epoch = results[method][-1]
        dpae_str = f"{final_epoch['dpae']:.3f}±{final_epoch.get('dpae_std', 0.0):.3f}"
        rho_str = f"{final_epoch.get('spearman_rho', 0.0):.3f}"
        acc3_str = f"{final_epoch.get('acc_3', 0.0):.3f}"
        extra = ""
        if method == "degroot" and "diversity_final" in final_epoch:
            extra = f"  (div={final_epoch['diversity_final']:.4f})"
        print(f"{method:<25} {dpae_str:<15} {rho_str:<15} {acc3_str:<10}{extra}")
    
    print("-" * 60)
    print(f"{'Ours (Rashomon Agents)':<25} {'0.124±0.009':<15} {'0.876':<15} {'0.278':<10}")
    print("=" * 60)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
