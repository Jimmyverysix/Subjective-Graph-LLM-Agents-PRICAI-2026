# Subjective-Graph LLM Agents

Official implementation of **[Subjective-Graph LLM Agents for Simulating Uncertainty in Classroom Social Perception](https://arxiv.org/abs/2603.20750)**.

## Overview

This repository implements a multi-agent framework for studying how uncertainty in classroom social perception affects academic-belief formation. Each student agent operates over an individualized subjective graph rather than a shared omniscient network. These graphs determine what social information an agent can retrieve, whom the agent can interact with, and how received messages are incorporated into its beliefs.

The implementation includes:

- **Subjective graph construction** from reported friendships, questionnaire-derived inferred ties, and worry-dependent view perturbations;
- **Graph-constrained retrieval** that limits each agent's social evidence to its perceived neighborhood;
- **LLM-mediated interaction** for generating messages and estimating trust;
- **Bayesian belief updates** over student-level academic beliefs;
- **Multi-seed evaluation, ablations, baselines, and visualization utilities** used by the experimental pipeline.

## Repository Structure

```text
EduAgent_OpenSource/
├── LICENSE
├── README.md
└── rashomon_agents/
    ├── config.yaml
    ├── requirements.txt
    ├── data/
    │   └── Social and Mental Health Survey Questionnaire.md
    └── src/
        ├── agent.py
        ├── baseline_comparison.py
        ├── clean_dataset.py
        ├── data_loader.py
        ├── main.py
        ├── rag.py
        ├── simulator.py
        ├── subjective_graph.py
        ├── visualize.py
        ├── visualize_ci.py
        └── visualize_multi_seed.py
```

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/Jimmyverysix/EduAgent_OpenSource.git
cd EduAgent_OpenSource/rashomon_agents

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Data Preparation

Participant-level survey responses, friendship nominations, and exam records are not distributed in this repository because they contain sensitive student information. The tracked questionnaire file documents the survey instrument only; it does not contain participant responses.

Authorized users should place the cleaned table at the default location:

```text
rashomon_agents/data/FINAL_full_plus_all_scores_cleaned.v2.csv
```

Alternatively, update `data.cleaned_csv` in `rashomon_agents/config.yaml`. The loader expects the following groups of variables:

| Variable group | Expected fields |
| --- | --- |
| Student context | `student_id`, `grade_code`, `class_code`, and optional gender information |
| Friendship nominations | `friend_1_id` through `friend_6_id` |
| Academic records | Six exam-score columns and their corresponding class-rank columns |
| Personality indicators | Extroversion, emotional stability, optimism, and impulsivity pairs |
| Worry indicators | One-hot responses ranging from never worried to very concerned |

By default, the pipeline retains classes with at least 30 students and students with complete six-exam temporal coverage. See `rashomon_agents/src/data_loader.py` for the exact column names and filtering logic.

## OpenRouter Configuration

The default configuration uses `meta-llama/llama-3.1-8b-instruct` through the OpenRouter-compatible API endpoint. Set the API key before running an experiment:

```bash
export OPENROUTER_API_KEY="your-api-key"
```

For Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY="your-api-key"
```

The project also loads environment variables from a local `.env` file. Provider endpoint, model name, sampling parameters, timeouts, and concurrency can be changed under `llm` in `config.yaml`. OpenRouter credentials are required for the experimental workflow reported by the paper.

## Running Experiments

Run all commands from `rashomon_agents/`.

### Configuration check

The dry run uses one class, one epoch, and one interaction round while retaining the same data and LLM pipeline:

```bash
python -m src.main --config config.yaml --dry-run --seeds 42
```

### Main experiment

```bash
python -m src.main \
  --config config.yaml \
  --seeds 42,43,44,45,46 \
  --output main
```

The default simulation uses six epochs, three rounds per epoch, and up to three interaction partners. These values can be changed in `config.yaml` or overridden with `--epochs`, `--rounds`, and `--workers`.

### Ablations

The main entry point exposes the paper's component ablations:

```bash
# Remove graph-constrained retrieval
python -m src.main --config config.yaml --seeds 42,43,44,45,46 --no-rag --output no_rag

# Replace individualized views with the shared graph
python -m src.main --config config.yaml --seeds 42,43,44,45,46 --no-subjective-graph --output no_subjective_graph

# Disable LLM-based trust estimation
python -m src.main --config config.yaml --seeds 42,43,44,45,46 --no-llm-trust --output no_llm_trust

# Disable LLM-generated messages
python -m src.main --config config.yaml --seeds 42,43,44,45,46 --no-llm-message --output no_llm_message
```

### Baselines

```bash
python -m src.baseline_comparison \
  --config config.yaml \
  --output output/baselines
```

Use the built-in help for plotting arguments:

```bash
python -m src.visualize --help
python -m src.visualize_multi_seed --help
python -m src.visualize_ci --help
```

## Configuration

The main settings are centralized in `rashomon_agents/config.yaml`.

| Section | Purpose |
| --- | --- |
| `data` | Input data path and encoding |
| `selection` | Class selection, coverage, and cohort filters |
| `rag` | Local retrieval result count |
| `llm` | Provider endpoint, model, API-key variable, sampling, retries, and workers |
| `simulation` | Epochs, rounds, partners, batch size, and graph perturbation scale |
| `output` | Output root directory |

Each experiment directory records the simulation controls, seeds, and dry-run status in `config_used.json`.

## Outputs

Runs are stored under:

```text
rashomon_agents/output/<timestamp>[_<label>]/
```

Depending on the command, the directory contains per-seed metrics, LLM usage statistics, multi-seed summaries, and the resolved run configuration. The principal summary files are:

```text
config_used.json
seed_<seed>/aggregate_metrics.json
seed_<seed>/llm_stats.json
multi_seed_summary.json
```

Use `--resume-dir <existing-output-directory>` to continue an interrupted run. Because LLM responses may vary across provider and model revisions, record the model identifier, configuration, run date, and random seeds when reporting new results. The default implementation uses a remote LLM endpoint and does not require a local GPU.

## Scope of This Release

This repository includes the simulation implementation, default configuration, questionnaire instrument, baseline implementations, and visualization utilities. It does not include participant-level data, cached LLM responses, or generated experimental outputs. Reproducing the paper's numerical results therefore requires authorized access to the study data and a configured LLM endpoint.

## Citation

If you use this repository, please cite:

```bibtex
@article{yang2026subjectivegraph,
  title   = {Subjective-Graph LLM Agents for Simulating Uncertainty in Classroom Social Perception},
  author  = {Yang, Jinming and Jiang, Xinyu and Jiao, Xinshan and Zhang, Xinping},
  journal = {arXiv preprint arXiv:2603.20750},
  year    = {2026},
  doi     = {10.48550/arXiv.2603.20750},
  url     = {https://arxiv.org/abs/2603.20750}
}
```

## License

This project is released under the [MIT License](LICENSE).
