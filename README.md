# Rashomon Set Agents

Implementation of LLM-driven multi-agent simulation framework for modeling epistemic uncertainty in social perception via "Rashomon Set" agents.

## Overview

This project implements an LLM-driven multi-agent probabilistic modeling framework to study how individual differences in subjective social perception (Rashomon effect) propagate through micro-level interactions, leading to macro-level academic differentiation and collective misjudgment.

## Key Features

- **Subjective Graph**: Constructs independent subjective social graphs for each agent, simulating cognitive biases in the real world
- **Retrieval-Augmented Generation (RAG)**: Agents access information through limited-scope retrieval with cognitive noise injection related to anxiety levels
- **Bayesian Belief Update**: Agents perform Bayesian updates using LLM-generated trust weights
- **Multi-Agent Simulation**: Simulates complete semester interactions to study the propagation mechanism of cognitive uncertainty

## Project Structure

```
EduAgent_OpenSource/
├── rashomon_agents/         # Main project code
│   ├── config.yaml          # Configuration file
│   ├── requirements.txt    # Python dependencies
│   ├── src/                 # Source code
│   └── data/                # Data directory
└── README.md                # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd rashomon_agents
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data

This project does not include data files by default (for privacy and size considerations). You need to:

1. Prepare cleaned CSV data files
2. Configure the data path in `config.yaml`

### 3. Run Experiments

```bash
# Quick test (dry-run)
python -m src.main --config config.yaml --dry-run

# Full experiment
python -m src.main --config config.yaml --seeds 42,43,44,45,46
```

## Data Description

The data used in this project includes:
- Basic student information (grade, class, gender, etc.)
- Social network questionnaire data (friend relationships, social perception, etc.)
- 6 exam score sequences

**Important**: For privacy protection, this repository does not include any raw data files. Users need to prepare data files that meet the format requirements.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Issues and Pull Requests are welcome.

## Disclaimer

This code is for research purposes only. When using this code, please ensure compliance with relevant data protection regulations and ethical guidelines.
