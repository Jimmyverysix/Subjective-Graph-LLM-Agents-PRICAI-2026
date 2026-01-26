# Rashomon Set Agents（罗生门集合智能体）
UAI 2026 Proposal Implementation：通过“罗生门集合”智能体对社会感知中的认知不确定性建模。

本目录既是可运行代码仓库，也是 `Proposal_2.tex` 的配套复现脚手架：你可以直接跑主实验/消融/外部基线，并生成论文中引用的图表文件。

## 项目结构（关键文件）

```
rashomon_agents_proposal/
├── Proposal_2.tex                 # 论文草稿（会引用 output/ 下的若干图表）
├── config.yaml                    # 主配置（数据路径、LLM、仿真参数）
├── requirements.txt               # Python 依赖
├── .gitignore                     # 精细控制 output/：只放行论文必需的小产物
├── data/
│   └── README.md                  # 数据放置说明（默认不随仓库发布）
└── src/
    ├── main.py                    # 主入口：主实验 + 多种子 + 消融
    ├── data_loader.py             # 数据加载与样本筛选（Full-Temporal / Social-Observed）
    ├── subjective_graph.py        # 主观图（Rashomon set）与 No-Subjective-Graph 消融
    ├── rag.py                     # 受限视野检索（含 No-RAG 消融）
    ├── agent.py                   # LLM 叙事生成/信任门控/贝叶斯融合
    ├── simulator.py               # 仿真主循环 + 指标计算
    ├── baseline_comparison.py     # 外部基线（含 DeGroot sweep）
    ├── visualize.py               # 单次运行的图 + results_table.tex
    ├── visualize_uai.py           # 多种子/消融：std band 版本图
    └── visualize_uai_ci.py        # 多种子/消融：bootstrap CI + 配对差异图
```

## 指标定义（与实现一致）

- **DPAE**：\(\text{DPAE}=1-\rho(\hat{R},R)\)（Spearman 相关的互补量）
- **Top-k**：\(\mathrm{Acc@k}=|\mathrm{Top}_k(\hat{R})\cap \mathrm{Top}_k(R)|/k\)
- **Unc(t)**：\(\text{Unc}(t)=\frac{1}{N^2}\sum_{j,k}\sigma_{jk}^{(t)}\)（实现为所有 belief 的 `sigma` 平均）
- **Strat(t)**：实现中记录为 `true_score_var`，实际计算的是**当期班级排名**的方差（见 `src/simulator.py`）

## 安装与环境

```bash
cd rashomon_agents_proposal
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据（默认不随仓库发布）

本项目默认不提交 `data/*.csv`（敏感/体积大）。你需要在 `config.yaml` 里把 `data.cleaned_csv` 指向你本地的清洗后 CSV。

如需从原始 CSV 生成清洗版本，可用：

```bash
python -m src.clean_dataset --input /path/to/FINAL_full_plus_all_scores_cleaned.csv --output data/FINAL_full_plus_all_scores_cleaned.v2.csv
```

## 运行主实验 / 消融（对应 Proposal）

```bash
# 端到端 sanity check（推荐）
python -m src.main --config config.yaml --dry-run

# 单种子完整模拟（会写入 output/<timestamp>/seed_<seed>/...）
python -m src.main --config config.yaml --seeds 42

# 多种子（会生成 multi_seed_summary.json；供 uai 可视化脚本读取）
python -m src.main --config config.yaml --seeds 42,43,44,45,46

# 消融
python -m src.main --config config.yaml --seeds 42,43,44 --no-rag
python -m src.main --config config.yaml --seeds 42,43,44 --no-subjective-graph
python -m src.main --config config.yaml --seeds 42,43,44 --no-llm-trust
```

### 可选：真实 LLM

```bash
export OPENROUTER_API_KEY="your-api-key"
```

不设置 key 会走 mock 模式（便于快速跑通流程，但不代表真实性能）。

## 生成 `Proposal_2.tex` 引用的图表

### 1) 多种子主结果/消融图（`output/uai_figures/*`）

```bash
python -m src.visualize_uai \
  --baseline output/<baseline_exp>/multi_seed_summary.json \
  --no_rag output/<no_rag_exp>/multi_seed_summary.json \
  --no_subj output/<no_subj_exp>/multi_seed_summary.json \
  --no_trust output/<no_trust_exp>/multi_seed_summary.json \
  --outdir output/uai_figures
```

### 2) bootstrap CI + 配对差异图（`output/uai_figures_ci/*`）

```bash
python -m src.visualize_uai_ci \
  --baseline output/<baseline_exp> \
  --no_rag output/<no_rag_exp> \
  --no_subj output/<no_subj_exp> \
  --no_trust output/<no_trust_exp> \
  --outdir output/uai_figures_ci \
  --bootstrap 20000
```

### 3) 单次运行的图 + LaTeX 表（`output/**/figures/*`）

```bash
python -m src.visualize --output_dir output/<single_run_exp> --save_dir output/<single_run_exp>/figures
```

会生成 `results_table.tex` 以及若干 PNG，供 `Proposal_2.tex` 直接 `\\input`/`\\includegraphics`。

### 4) DeGroot sweep 图（`output/degroot_sweep/degroot_sweep_epoch6.png`）

```bash
python -m src.baseline_comparison \
  --config config.yaml \
  --degroot-sweep-steps "1,5,30,100" \
  --degroot-sweep-epoch 6 \
  --output output/degroot_sweep \
  --degroot-sweep-only
```

## 关于提交/上传（重要）

- `output/` 里会产生**大量 cache 与大体积 JSON**（尤其是 `output/**/cache/`、`seed_*/class_*_metrics.json` 等）。
- 本项目通过 `rashomon_agents_proposal/.gitignore` 做了“白名单放行”：只允许提交论文引用到的小图/小表格文件，其他默认忽略。

## 引用

```bibtex
@inproceedings{anonymous2026rashomon,
  title={Modeling Epistemic Uncertainty in Social Perception via "Rashomon Set" Agents},
  author={Anonymous},
  booktitle={Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI)},
  year={2026}
}
```
