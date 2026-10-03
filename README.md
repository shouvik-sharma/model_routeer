# Prompt complexity classifier

Scores a prompt on 14 routing parameters, computes a weighted complexity score, and classifies a **nano / super / ultra** model tier.

## Setup

```powershell
pip install pandas numpy scikit-learn
$env:PYTHONPATH = "d:\QuantPath"
```

## Train and classify

```powershell
python -m tools.prompt_classifier --train --n 800
python -m tools.prompt_classifier --prompt "Compare SMA crossover vs RSI and pick one."
python examples/classify_prompt_demo.py
```
