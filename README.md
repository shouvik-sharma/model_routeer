# Model Router & Prompt Complexity Classifier

An intelligent prompt classification and LLM routing system. It scores incoming user prompts against a **14-parameter rubric**, computes a weighted complexity score, and dynamically routes prompts to the most cost-effective model tier: **Nano**, **Super**, or **Ultra**.

---

## Architecture & Routing Tiers

| Tier | Default Model | Typical Workload & Task Types |
| :--- | :--- | :--- |
| **Nano** | `nemotron-nano` | Factual Q&A, short summarization, basic lookup, greetings, low-ambiguity tasks |
| **Super** | `nemotron-super` | Code generation, multi-step math/logic, tool execution, structural transformation |
| **Ultra** | `nemotron-ultra` | Complex research, strategic planning, high-accuracy formal proofs, open-ended architecture |

### 14 Rubric Scoring Parameters
* **Reasoning Complexity** (20%)
* **Task Type** (15%)
* **Decision Search Complexity** (12%)
* **Accuracy Requirement** (10%)
* **Instruction Complexity** (9%)
* **Context Requirement** (8%)
* **Output Complexity** (7%)
* **Iteration / Verification Need** (6%)
* **Domain Difficulty** (4%)
* **Tool Requirement** (3%)
* **Freshness Requirement** (2%)
* **Ambiguity** (2%)
* **Latency Sensitivity** (1%)
* **Cost Sensitivity** (1%)

---

## Project Structure

```text
QuantPath/
├── .venv/                         # Local Python virtual environment
├── cache/
│   └── models/
│       └── prompt_router.pkl      # Pre-trained classifier weights
├── data/
│   ├── arena/                     # LMSYS Chatbot Arena datasets
│   │   ├── arena_prompts_english.parquet   (23,613 rows, 5.5 MB)
│   │   └── arena_prompts_english.csv       (23,613 rows, 23.9 MB)
│   └── routerbench/               # Martian RouterBench datasets
│       ├── routerbench_0shot_english.parquet (392,832 rows, 137.3 MB)
│       ├── routerbench_0shot_long.parquet    (401,467 rows, 138.2 MB)
│       └── routerbench_5shot_long.parquet    (401,621 rows, 213.5 MB)
├── examples/
│   └── classify_prompt_demo.py    # Demo script showing training & inference
├── tools/
│   ├── prompt_classifier.py       # Core 14-parameter extractor & classifier
│   ├── get_routerbench.py         # RouterBench downloader & transformer
│   └── get_arena_prompts.py       # LMSYS Chatbot Arena extractor & cleaner
├── prompt.ipynb                   # Interactive Jupyter notebook
├── requirements.txt               # Dependencies
└── README.md
```

---

## Quickstart & Environment Setup

### 1. Create Virtual Environment
Using [uv](https://github.com/astral-sh/uv) (recommended) or standard Python:

```powershell
# Using uv:
uv venv .venv --python 3.11
.venv\Scripts\activate

# Or using standard python:
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Register Jupyter Kernel
To execute notebooks in VS Code / Antigravity IDE without environment warnings:
```powershell
python -m ipykernel install --user --name "quantpath-venv" --display-name "Python (.venv)"
```

---

## Datasets

The repository includes utilities to fetch and prepare two datasets:

### 1. LMSYS Chatbot Arena Conversations (Real User Prompts)
Best for training the prompt categorization and ambiguity layer. Contains real-world human prompts, model pairings, and pairwise preference votes (`winner`).

* **Dataset ID:** [`lmsys/chatbot_arena_conversations`](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) *(Gated dataset)*
* **Prerequisite:** Accept license agreement on Hugging Face and obtain a free Read Token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
* **Download & Process:**
  ```powershell
  python -m tools.get_arena_prompts --token YOUR_HF_TOKEN
  # Or with environment variable:
  $env:HF_TOKEN = "YOUR_HF_TOKEN"
  python -m tools.get_arena_prompts
  ```
* **Output:**
  * `data/arena/arena_prompts_english.parquet` (23,613 unique English prompts)
  * `data/arena/arena_prompts_english.csv`

### 2. Martian RouterBench (Model Performance Benchmarks)
Best for evaluating router performance, accuracy vs. cost trade-offs, and latency optimization.

* **Dataset ID:** [`withmartian/routerbench`](https://huggingface.co/datasets/withmartian/routerbench)
* **Download & Process:**
  ```powershell
  # Download 0-shot benchmark (401k rows in long format):
  python -m tools.get_routerbench --split 0shot --format long --save-parquet data/routerbench/routerbench_0shot_long.parquet

  # Or download 5-shot benchmark:
  python -m tools.get_routerbench --split 5shot --format long --save-parquet data/routerbench/routerbench_5shot_long.parquet
  ```
* **Pre-generated Subsets:**
  * `data/routerbench/routerbench_0shot_english.parquet` (392,832 rows, filtered for English-only benchmarks like GSM8K, MBPP, Hellaswag, MMLU).

---

## Usage Examples

### 1. CLI Prompt Classification
Classify any prompt directly from the terminal:

```powershell
python -m tools.prompt_classifier --prompt "Write a Python script using pandas to backtest a 20/50 SMA crossover strategy on SPY."
```

Output:
```text
PROMPT: Write a Python script using pandas to backtest a 20/50 SMA crossover strategy on SPY.
Weighted Complexity Score: 0.612
Route Tier:        SUPER
Recommended Model: nemotron-super
Task Type:         code_generation
```

### 2. Python API
```python
from tools.prompt_classifier import classify_prompt

result = classify_prompt("Design an automated order routing microservice in Python with asyncio and redis.")
print(f"Tier:  {result.route_tier.upper()}")
print(f"Model: {result.recommended_model}")
print(f"Score: {result.weighted_score:.3f}")

# Access full 14-parameter rubric scores:
scores = result.to_dict()["scores"]
print(f"Reasoning Complexity: {scores['reasoning_complexity']}")
print(f"Task Type:            {scores['predicted_task_type']}")
```

### 3. Loading Prepared Datasets in Python
```python
import pandas as pd

# Load 23.6K English real user prompts with preference votes:
df_arena = pd.read_parquet("data/arena/arena_prompts_english.parquet")

# Load 392K English benchmark prompts (MBPP, GSM8K, MMLU, etc.):
df_routerbench = pd.read_parquet("data/routerbench/routerbench_0shot_english.parquet")
```

### 4. Interactive Jupyter Notebook
Open [`prompt.ipynb`](file:///D:/QuantPath/prompt.ipynb) in your IDE:
1. Click **Select Kernel** in the top right.
2. Choose **`Python (.venv)`**.
3. Click **Run All** to preview English benchmark data and test prompt routing interactively.
