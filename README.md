# Rashomon Set Agents（罗生门集合智能体）

UAI 2026 Proposal Implementation：通过"罗生门集合"智能体对社会感知中的认知不确定性建模。

## 项目简介

本项目实现了一个 LLM 驱动的多智能体概率建模框架，用于研究个体主观社交感知的差异（罗生门效应）如何通过微观交互传播，进而涌现宏观学业分化与集体误判。

## 主要特性

- **主观图谱（Subjective Graph）**：为每个智能体构建独立的主观社交图谱，模拟真实世界中的认知偏差
- **检索增强生成（RAG）**：智能体通过受限视野检索访问信息，注入与焦虑水平相关的认知噪声
- **贝叶斯信念更新**：智能体通过 LLM 生成的可信度权重进行贝叶斯更新
- **多智能体仿真**：模拟完整学期的交互过程，研究认知不确定性的传播机制

## 项目结构

```
EduAgent_OpenSource/
├── rashomon_agents_proposal/    # 主项目代码
│   ├── Proposal_2.tex          # 论文草稿
│   ├── config.yaml             # 配置文件
│   ├── requirements.txt        # Python 依赖
│   ├── src/                    # 源代码
│   └── data/                   # 数据目录（需自行准备）
└── README.md                   # 本文件
```

详细的项目结构和使用说明请参考 [rashomon_agents_proposal/README.md](rashomon_agents_proposal/README.md)。

## 快速开始

### 1. 安装依赖

```bash
cd rashomon_agents_proposal
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 准备数据

本项目默认不包含数据文件（出于隐私和体积考虑）。您需要：

1. 准备清洗后的 CSV 数据文件
2. 在 `config.yaml` 中配置数据路径

### 3. 运行实验

```bash
# 快速测试（dry-run）
python -m src.main --config config.yaml --dry-run

# 完整实验
python -m src.main --config config.yaml --seeds 42,43,44,45,46
```

## 数据说明

本项目使用的数据包含：
- 学生基本信息（年级、班级、性别等）
- 社交网络问卷数据（好友关系、社交感知等）
- 6 次考试成绩序列

**重要**：出于隐私保护，本仓库不包含任何原始数据文件。用户需要自行准备符合格式要求的数据。

## 引用

如果您使用本代码，请引用：

```bibtex
@inproceedings{anonymous2026rashomon,
  title={Modeling Epistemic Uncertainty in Social Perception via "Rashomon Set" Agents},
  author={Anonymous},
  booktitle={Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI)},
  year={2026}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request。

## 免责声明

本代码仅供研究使用。使用本代码时，请确保遵守相关数据保护法规和伦理准则。
