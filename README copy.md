# AVI Research

Research repository for evaluating **AVI (AI Validation Interface)** on FinanceBench dataset.

## 🎯 Purpose

Experiment for paper: **"Dynamic Bilateral Alignment for Enterprise AI Governance"**

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
vim .env

# 3. Download FinanceBench
python scripts/01_download_financebench.py

# 4. Transform dataset
python scripts/02_transform_dataset.py

# 5. Upload to AVI and run experiment
# (See notebooks/ for interactive workflow)
```

## 📁 Structure

- `src/` - Python modules (transform, experiment, visualization)
- `scripts/` - CLI scripts for automation
- `notebooks/` - Jupyter notebooks (full workflow)
- `config/` - Configuration files
- `data/` - Data files (gitignored)
- `paper/` - Generated figures and tables

## 📊 Modules

### Transform (`src/transform/`)
- `policy_generator.py` - LLM-based embargo policy generation
- `context_generator.py` - Alternative context generation
- `dataset_builder.py` - Complete dataset builder

### Experiment (`src/experiment/`)
- `evaluator.py` - Automatic metrics
- `llm_judge.py` - LLM-as-a-Judge evaluation

### Utils (`src/utils/`)
- `llm_client.py` - Unified LLM client (OpenAI, Cotype)
- `helpers.py` - Helper utilities

## 📜 License

MIT License
