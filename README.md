# Computational Linguistics Project

This project implements a multilingual text analysis system using Large Language Models (LLMs) to detect irony, misogyny, and stance in social media posts. It compares zero-shot and few-shot learning approaches across different languages.

## Features

- Analysis of three text categories:
  - Misogyny detection (Italian, English)
  - Irony detection (Italian, English)
  - Stance detection (Italian, French)
- Comparison of two LLM approaches:
  - Zero-shot learning
  - Few-shot learning with 3 examples
- Multiple model support:
  - Llama2-uncensored
  - Llama3.2
  - DeepSeek (for evaluation)
- Comprehensive result analysis with Excel output

## Setup

1. Install dependencies:
```bash
pip install ollama pandas xlsxwriter
```

2. Ensure the following directory structure:
```
.
├── data/
│   └── processed/
│       ├── irony_combined.csv
│       ├── misogyny_combined.csv
│       └── stance_combined.csv
├── scripts/
│   └── main.py
└── results/
```

3. Install and configure Ollama with required models:
```bash
ollama pull llama2-uncensored
ollama pull llama3.2
ollama pull deepseek-r1:8b
```

## Usage

Run the main script:
```bash
python scripts/main.py
```

Results will be saved in `results/experiment_results_[timestamp].xlsx` with multiple sheets:
- All Results: Raw data from all experiments
- Category-specific sheets (Irony, Misogyny, Stance)
- Language comparison sheets for each category

## Data Sources

- Misogyny: AMI 2018 dataset (Italian/English tweets)
- Irony: MultiPiCo corpus (multilingual Twitter/Reddit conversations)
- Stance: x-stance dataset (German/French/Italian political comments)