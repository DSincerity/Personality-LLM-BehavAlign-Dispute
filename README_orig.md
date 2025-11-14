# Behavioral Alignment between LLMs and Humans in Dispute Scenario

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://aaai.org/conference/aaai/aaai-26/aisi-call/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation of our **AAAI-2026 paper** on **personality-driven behavioral alignment in conflict dialogue**, providing code and evaluation scripts to compare Big Five–prompted LLM agents with humans in dispute resolution conversations.

## 📄 Paper Information

**Title:** [Evaluating Behavioral Alignment in Conflict Dialogue: A Multi-Dimensional Comparison of LLM Agents and Humans](TBD)

**Authors:** Deuksin Kwon, Kaleen Shrestha, Spencer Lin, James Hale, Jonathan Gratch, Maja Matarić, Gale Lucas

**Conference:** AAAI 2026: Special Track on Artificial Intelligence for Social Impact (AISI)
---
<div align="center">
<img src="resources/intro_fig.png" alt="Behavioral Alignment Evaluation Framework Overview" width="350"/>
</div>

## Abstract

Large language models (LLMs) are increasingly used to simulate human behavior in social settings such as legal mediation, negotiation, and dispute resolution. However, it remains unclear if LLMs simulating human behavior are consistent with theoretically proposed mechanisms. Human personality, for instance, may shape how individuals navigate social interactions, including strategic choices and behaviors in emotionally charged interactions. This raises a critical question: \textit{Can LLMs, when prompted with personality traits, reproduce personality-driven differences in human conflict behavior?} To explore this, we introduce an evaluation framework that enables direct comparison of human-human and LLM-LLM behaviors in dispute resolution dialogues with respect to Big Five Inventory (BFI) personality traits. This framework introduces a set of interpretable metrics related to strategic behavior and conflict outcomes. We additionally contribute a novel dataset creation methodology for LLM dispute resolution dialogs with matched scenarios and personality traits with respect to human conversations. Finally, we demonstrate the usage of our evaluation framework with three contemporary closed-source LLMs and show significant divergences in how personality manifests in conflict across different LLMs compared to human data, challenging the assumption that personality-prompted agents can serve as reliable behavioral proxies in socially impactful applications. Our work highlights the need for psychological grounding and rigorous validation in AI simulations before real-world use.

---
## Supplementary Materials of the Paper
AAAI_2026_Supplementary Material.pdf

---
Technical details and code to be released soon before the AAAI 2026 conference.
## 1. Overview

This repository provides a complete pipeline to:

1. **simulate** LLM–LLM conflict/negotiation dialogues,
2. **annotate** them on with IRP strategies
3. **evaluate** behavioral alignment between LLM agents and humans on strategic dimensions in terms of personality-driven behavior.


---

## 3. Setup

```bash
# Install dependencies (Python 3.10 recommended)
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

**Supported model IDs (scripts)**

- `gpt-4.1-mini`
- `gpt-4.1`
- `claude-3-7-sonnet-20250219`
- `gemini-2.0-flash`

*Use the models your account actually has access to.*

## 4. Data Generation Pipelines

This repository contains two parallel pipelines:
1.	LLM-to-LLM (simulated) pipeline — to create model dialogues.
2.	KODIS (human) pipeline — to create/refresh the human baseline.

Keeping them separate makes it clear which data goes into which side of the comparison.

### 4.1 LLM-to-LLM Pipeline
Generate LLM negotiation data with multi-level behavioral annotations
**Pipeline**:
```
┌─────────────────────────────────────────────────────────────────┐
│  Simulation (GPT-4/Claude/Gemini)                               │
│  ↓ Generate N negotiation conversations                       │
│  data/simulations/{model}.json                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Emotion Annotation (EmoBERTa)                                  │
│  ↓ Label each utterance with 7 emotion classes                  │
│  data/emotions/{model}_emo.json                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  IRP Strategy Annotation (GPT-4 + Majority Voting)              │
│  ↓ Label each utterance with 9 IRP strategy types               │
│  data/complete/{model}_complete.json                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Linguistic Entrainment Calculation (nCLiD + Word2Vec)          │
│  ↓ Calculate LE values for coordination patterns                │
│  data/linguistic_entrainment/LE_values_{model}.csv              │
└─────────────────────────────────────────────────────────────────┘
           ↓ All data files ready for evaluation
```

```bash
# Full pipeline (recommended)
./scripts/generate_data.sh --model gpt-4.1-mini --n-exp 250

# Step-by-step execution
python scripts/run_simulation.py --agent_1_engine gpt-4.1-mini --agent_2_engine gpt-4.1-mini --n_exp 250
python scripts/annotate_emotions.py --model gpt-4.1-mini --input data/simulations/gpt-4.1-mini.json
python scripts/annotate_irp.py --model-name gpt-4.1-mini --input data/emotions/gpt-4.1-mini_emo.json
python scripts/merge_irp.py --model-name gpt-4.1-mini --output data/complete/gpt-4.1-mini_complete.json
python scripts/calculate_entrainment.py --model gpt-4.1-mini --input data/complete/gpt-4.1-mini_complete.json

# Resume from checkpoint (e.g., simulation done, annotation failed):
./scripts/generate_data.sh --model gpt-4.1-mini --skip-simulation
```

**Supported Models**: `gpt-4.1-mini`, `gpt-4.1`, `claude-3-7-sonnet-20250219`, `gemini-2.0-flash`

### 4.2 KODIS Human Baseline Annotation

```bash
# Emotion annotation
python scripts/annotate_emotions.py \
    --input data/KODIS/KODIS_merged_20_samples.json \
    --data-type kodis \
    --output data/KODIS_merged_20_samples_emo.json

# IRP annotation (5x majority voting for higher accuracy)
python scripts/annotate_irp.py \
    --input data/KODIS/KODIS_merged_20_samples_emo.json \
    --output-dir IRP_Annotation/KODIS_annotations \
    --data-type kodis \
    --majority-voting 5

python scripts/merge_irp.py \
    --input data/KODIS/KODIS_merged_20_samples_emo.json \
    --annotation-dir IRP_Annotation/KODIS_annotations \
    --output data/KODIS/KODIS_merged_20_samples_emo_irp.json \
    --data-type kodis

# Linguistic Entrainment calculation
python scripts/calculate_entrainment.py \
    --input data/KODIS/KODIS_merged_20_samples_emo_irp.json \
    --data-type kodis
```

## 5. Run Behavioral Alignment Evaluation
After you have generated all required data files:
- **Human data**: `data/KODIS/..._emo_irp.json` (required)
- **Agent data**: `data/complete/*_complete.json` (required)
- **Linguistic Entrainment**: `data/linguistic_entrainment/LE_values_{model}.csv` (required, for LEG metric)
- **KODIS Entrainment**: `data/linguistic_entrainment/LE_values_KODIS.csv` (required, for LEG metric)
- **LIWC files**: `data/LIWC/LIWC_22_Aggregated_{model}.csv` (optional, for LG metrics)

you can run the evaluator.

### 5.1 Auto-detect All Models
```bash
#Auto-detect all models (scans data/complete/ directory)
./scripts/run_evaluation.sh --auto-detect
# → Scans: data/complete/ for *_complete.json files for models
# → Scans: data/KODIS/ for KODIS_merged_20_samples_emo_irp.json for human data
# → Evaluates: gpt-4.1-mini, claude-3-7-sonnet-20250219, etc.
# → LIWC directory: data/LIWC/ (default)
# Output: evaluation_results/evaluation_report_*.md
```

### 5.2 Specify Models and Parameters
```bash
python scripts/run_evaluation.py \
    --kodis_emo_path data/KODIS/KODIS_merged_20_samples_emo_irp.json \
    --kodis_irp_path data/KODIS/KODIS_merged_20_samples_emo_irp.json \
    --agent_data_dir data/complete \
    --models gpt-4.1 gpt-4.1-mini claude-3-7-sonnet-20250219 gemini-2.0-flash
```

**Key Parameters:**
- `--kodis_emo_path`: Path to KODIS emotion-annotated data
- `--kodis_irp_path`: Path to KODIS IRP-annotated data
- `--agent_data_dir`: Directory containing agent simulation results (default: `data/complete`)
- `--models`: List of model names to evaluate
  - **Default**: All models (`gpt-4.1`, `gpt-4.1-mini`, `claude-3-7-sonnet-20250219`, `gemini-2.0-flash`)
  - Specify specific models to evaluate only those (e.g., `--models gpt-4.1 claude-3-7-sonnet-20250219`)
- `--output_dir`: Output directory (default: `evaluation_results/`)
- `--liwc_dir`: Directory containing LIWC CSV files (default: `data/LIWC/`)
- `--metrics`: Select specific metrics to run (e.g., `--metrics anger strategic`)
  - **Available options**:
    - `anger`: Anger Trajectory Analysis (DTW, AUC, ATG, AMG)
    - `strategic`: Strategic Behavior Analysis (IRP distribution, JSD, SBG)
    - `linguistic`: Linguistic Gap Analysis (LG-Dispute, LG-IRP)
    - `entrainment`: Linguistic Entrainment Gap (LEG) using nCLiD
    - `all`: Run all available metrics (default)
  - Can specify multiple metrics separated by space (e.g., `--metrics anger strategic`)
- `--use_cache`: Use cached result files if available (default: `True`)
- `--no_cache`: Force recomputation, ignore cached result files
- `--skip_anger`: (Deprecated) Skip anger trajectory analysis
- `--skip_strategic`: (Deprecated) Skip strategic behavior analysis
- `--skip_linguistic`: (Deprecated) Skip linguistic gap analysis
- `--skip_entrainment`: (Deprecated) Skip linguistic entrainment analysis

**Outputs**:
- Markdown reports in evaluation_results/
- Scores for: ATG, AMG, SBG, LEG
- LIWC-based scores (LG-Dispute / LG-IRP) if LIWC is available


## 6. LIWC Setup (Optional)

LIWC files enable **LG-Dispute** and **LG-IRP** metrics. Other metrics (ATG, AMG, SBG, LEG) work without LIWC.

**Required files**:
```
data/LIWC/
├── LIWC_22_Aggregated_KODIS.csv
└── LIWC_22_Aggregated_{model}.csv
```

**How to generate**:
1. Purchase [LIWC-22 software](https://www.liwc.app/)
2. Export conversation texts from data files
3. Run LIWC-22 analysis
4. Save aggregated CSV to `data/LIWC/`

**File format example**: `data/LIWC/LIWC_22_Aggregated_KODIS_sample.csv` (10-row sample)
**If LIWC missing**: Evaluation auto-skips LG metrics, computes ATG/AMG/SBG/LEG normally.

---

## 7. Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kwon-etal-2025-evaluating,
    title = "Evaluating Behavioral Alignment in Conflict Dialogue: A Multi-Dimensional Comparison of {LLM} Agents and Humans",
    author = "Kwon, Deuksin  and Shrestha, Kaleen  and Han, Bin  and Lee, Elena Hayoung  and Lucas, Gale",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.828/",
    pages = "16377--16391",
    ISBN = "979-8-89176-332-6",
}
```
---

## Contact

For questions or issues:
- **GitHub Issues**: Report bugs or feature requests
- **Email**: deuksink@usc.edu (Brian Deuksin Kwon)

---

## License
This project is licensed under the MIT License.
