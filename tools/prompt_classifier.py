"""
Prompt complexity classifier for LLM routing.

Scores a user prompt on 14 rubric parameters, computes a weighted complexity
score, and classifies a routing tier (nano / super / ultra).

Two layers:
  1. Interpretable heuristic scorers (works with no training data).
  2. sklearn RandomForest on the 14 scores, trained on synthetic labels
     so the mapping from scores -> tier can be fit / replaced with real labels.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("cache/models/prompt_router.pkl")

WEIGHTS: dict[str, float] = {
    "reasoning_complexity": 0.20,
    "task_type": 0.15,
    "decision_search_complexity": 0.12,
    "accuracy_requirement": 0.10,
    "instruction_complexity": 0.09,
    "context_requirement": 0.08,
    "output_complexity": 0.07,
    "iteration_verification_need": 0.06,
    "domain_difficulty": 0.04,
    "tool_requirement": 0.03,
    "freshness_requirement": 0.02,
    "ambiguity": 0.02,
    "latency_sensitivity": 0.01,
    "cost_sensitivity": 0.01,
}

TASK_TYPES = (
    "classification",
    "extraction",
    "generation",
    "coding",
    "analysis",
    "planning",
    "research",
    "mathematical_reasoning",
)

# How much inherent difficulty each task type contributes to the 15% slot.
TASK_TYPE_LOAD: dict[str, float] = {
    "classification": 0.25,
    "extraction": 0.30,
    "generation": 0.45,
    "research": 0.65,
    "analysis": 0.70,
    "coding": 0.80,
    "planning": 0.85,
    "mathematical_reasoning": 0.92,
}

ROUTE_MODELS = {
    "nano": "nemotron-nano",
    "super": "nemotron-super",
    "ultra": "nemotron-ultra",
}

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _hit_ratio(text: str, patterns: Iterable[str]) -> float:
    lowered = text.lower()
    hits = sum(1 for p in patterns if p in lowered)
    return min(1.0, hits / max(1, len(list(patterns)) * 0.25))


def _count_hits(text: str, patterns: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for p in patterns if p in lowered)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class ParameterScores:
    reasoning_complexity: float
    task_type: float
    decision_search_complexity: float
    accuracy_requirement: float
    instruction_complexity: float
    context_requirement: float
    output_complexity: float
    iteration_verification_need: float
    domain_difficulty: float
    tool_requirement: float
    freshness_requirement: float
    ambiguity: float
    latency_sensitivity: float
    cost_sensitivity: float
    predicted_task_type: str


@dataclass
class ClassificationResult:
    prompt: str
    scores: ParameterScores
    weighted_score: float
    route_tier: str
    recommended_model: str
    rationale: dict = field(default_factory=dict)
    source: str = "heuristic"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["scores"] = asdict(self.scores)
        return payload


class PromptFeatureExtractor:
    """Map raw prompt text onto the 14 rubric dimensions (0-1 each)."""

    REASONING_CUES = (
        "step by step",
        "reason",
        "because",
        "therefore",
        "trade-off",
        "tradeoff",
        "optimize",
        "optimisation",
        "optimization",
        "constraint",
        "if and only if",
        "depends",
        "implications",
        "derive",
        "prove",
        "why",
        "how would",
        "multi-step",
        "chain",
        "first then",
        "consider both",
    )
    DECISION_CUES = (
        "compare",
        "versus",
        " vs ",
        "alternatives",
        "option a",
        "choose",
        "which is better",
        "debate",
        "pros and cons",
        "rank",
        "select the best",
        "search",
        "enumerate",
        "candidates",
        "decision",
    )
    ACCURACY_CUES = (
        "exact",
        "precisely",
        "must be correct",
        "do not hallucinate",
        "cite",
        "source",
        "production",
        "legal",
        "compliance",
        "must not",
        "highly reliable",
        "zero error",
        "unit test",
        "numerically",
        "to the nearest",
    )
    INSTRUCTION_CUES = (
        "must",
        "should",
        "never",
        "always",
        "except",
        "unless",
        "only if",
        "priority",
        "constraints:",
        "rules:",
        "do not",
        "if ",
        "otherwise",
        "when ",
    )
    OUTPUT_CUES = (
        "json",
        "yaml",
        "table",
        "markdown",
        "report",
        "csv",
        "schema",
        "function",
        "class ",
        "code",
        "multi-part",
        "sections",
        "bullet",
        "template",
        "diff",
    )
    VERIFY_CUES = (
        "verify",
        "validate",
        "double-check",
        "self-check",
        "critique",
        "review",
        "refine",
        "iterate",
        "test cases",
        "sanity check",
        "cross-check",
        "proofread",
        "edge case",
    )
    DOMAIN_CUES = (
        "black-scholes",
        "option greeks",
        "implied volatility",
        "sharpe",
        "drawdown",
        "backtest",
        "quant",
        "stochastic",
        "kalman",
        "sec filing",
        "derivative",
        "bayesian",
        "likelihood",
        "jurisdiction",
        "statute",
        "pharmaco",
        "genome",
        "finite element",
    )
    TOOL_CUES = (
        "web search",
        "google",
        "browse",
        "api",
        "database",
        "sql",
        "execute",
        "run code",
        "python",
        "calculator",
        "fetch",
        "file",
        "csv",
        "endpoint",
        "sandbox",
    )
    FRESHNESS_CUES = (
        "today",
        "latest",
        "current",
        "realtime",
        "real-time",
        "as of",
        "this week",
        "breaking",
        "live",
        "now",
        "2025",
        "2026",
        "yesterday",
    )
    AMBIGUITY_CUES = (
        "something",
        "stuff",
        "somehow",
        "maybe",
        "whatever",
        "you know",
        "kind of",
        "etc",
        "idk",
        "help me with this",
        "fix it",
        "make it better",
    )
    LATENCY_CUES = (
        "quickly",
        "asap",
        "fast",
        "brief",
        "short answer",
        "tl;dr",
        "in one sentence",
        "low latency",
        "realtime response",
    )
    COST_CUES = (
        "cheap",
        "low cost",
        "minimize tokens",
        "short",
        "concise",
        "budget",
        "nano",
        "don't overthink",
        "simple answer",
    )
    TASK_PATTERNS: dict[str, tuple[str, ...]] = {
        "classification": ("classify", "label", "category", "sentiment", "is this"),
        "extraction": ("extract", "parse", "pull out", "named entit", "fields from"),
        "generation": ("write", "draft", "compose", "generate", "story", "email"),
        "coding": ("code", "python", "function", "implement", "debug", "refactor", "script"),
        "analysis": (
            "analyze",
            "analyse",
            "explain why",
            "diagnose",
            "interpret",
            "metrics",
            "compare",
            "versus",
            "trade-off",
            "pick one",
            "pros and cons",
        ),
        "planning": ("plan", "roadmap", "schedule", "steps to", "architecture", "design a"),
        "research": ("research", "survey", "literature", "what is known", "find sources"),
        "mathematical_reasoning": (
            "prove",
            "derive",
            "equation",
            "integral",
            "probability",
            "optimize the",
            "closed form",
            "calculate",
        ),
    }

    def extract(self, prompt: str) -> ParameterScores:
        text = prompt.strip()
        tokens = _tokens(text)
        n_tokens = max(1, len(tokens))
        n_chars = max(1, len(text))
        sentences = max(1, len(re.split(r"[.!?]+", text)))
        numbered = len(re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]|[-*])\s+", text))
        conditionals = len(re.findall(r"\b(if|unless|except|otherwise|when|provided that)\b", text, re.I))

        predicted_task = self._predict_task_type(text)

        reasoning = _clip01(
            0.35 * _hit_ratio(text, self.REASONING_CUES)
            + 0.20 * min(1.0, conditionals / 4)
            + 0.20 * min(1.0, sentences / 8)
            + 0.15 * min(1.0, n_tokens / 180)
            + 0.10 * (1.0 if predicted_task in {"planning", "mathematical_reasoning", "coding", "analysis"} else 0.2)
        )
        decision = _clip01(
            0.55 * _hit_ratio(text, self.DECISION_CUES)
            + 0.25 * min(1.0, text.lower().count(" or ") / 3)
            + 0.20 * min(1.0, numbered / 5)
        )
        accuracy = _clip01(
            0.50 * _hit_ratio(text, self.ACCURACY_CUES)
            + 0.25 * min(1.0, len(re.findall(r"\d+(?:\.\d+)?", text)) / 8)
            + 0.25 * (1.0 if predicted_task in {"mathematical_reasoning", "coding", "extraction"} else 0.3)
        )
        instruction = _clip01(
            0.40 * min(1.0, _count_hits(text, self.INSTRUCTION_CUES) / 6)
            + 0.30 * min(1.0, numbered / 6)
            + 0.30 * min(1.0, conditionals / 5)
        )
        context = _clip01(
            0.45 * min(1.0, n_tokens / 250)
            + 0.25 * min(1.0, n_chars / 1600)
            + 0.30 * (1.0 if "following" in text.lower() or "context:" in text.lower() else 0.15)
        )
        output = _clip01(
            0.70 * _hit_ratio(text, self.OUTPUT_CUES)
            + 0.30 * (1.0 if predicted_task in {"coding", "generation", "planning"} else 0.2)
        )
        iteration = _clip01(0.85 * _hit_ratio(text, self.VERIFY_CUES) + 0.15 * reasoning)
        domain = _clip01(
            0.75 * _hit_ratio(text, self.DOMAIN_CUES)
            + 0.25 * (1.0 if predicted_task in {"mathematical_reasoning", "analysis"} else 0.2)
        )
        tools = _clip01(_hit_ratio(text, self.TOOL_CUES))
        freshness = _clip01(_hit_ratio(text, self.FRESHNESS_CUES))
        # Ambiguity is high when the prompt is short/vague and low when constraints are dense.
        ambiguity = _clip01(
            0.40 * _hit_ratio(text, self.AMBIGUITY_CUES)
            + 0.35 * (1.0 - min(1.0, n_tokens / 40))
            + 0.25 * (1.0 if n_tokens < 12 else 0.1)
            - 0.35 * instruction
        )
        latency = _clip01(_hit_ratio(text, self.LATENCY_CUES))
        cost = _clip01(_hit_ratio(text, self.COST_CUES) * 0.8 + 0.2 * latency)

        return ParameterScores(
            reasoning_complexity=round(reasoning, 4),
            task_type=round(TASK_TYPE_LOAD[predicted_task], 4),
            decision_search_complexity=round(decision, 4),
            accuracy_requirement=round(accuracy, 4),
            instruction_complexity=round(instruction, 4),
            context_requirement=round(context, 4),
            output_complexity=round(output, 4),
            iteration_verification_need=round(iteration, 4),
            domain_difficulty=round(domain, 4),
            tool_requirement=round(tools, 4),
            freshness_requirement=round(freshness, 4),
            ambiguity=round(_clip01(ambiguity), 4),
            latency_sensitivity=round(latency, 4),
            cost_sensitivity=round(cost, 4),
            predicted_task_type=predicted_task,
        )

    def _predict_task_type(self, text: str) -> str:
        scores = {
            name: _count_hits(text, cues) for name, cues in self.TASK_PATTERNS.items()
        }
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            if len(_tokens(text)) < 20:
                return "generation"
            return "analysis"
        return best


class HeuristicRouter:
    """Weighted sum of the 14 scores, with routing knobs for latency/cost."""

    def __init__(self, extractor: PromptFeatureExtractor | None = None):
        self.extractor = extractor or PromptFeatureExtractor()

    def weighted_score(self, scores: ParameterScores) -> float:
        data = asdict(scores)
        total = 0.0
        for name, weight in WEIGHTS.items():
            total += weight * float(data[name])
        # Latency/cost are routing penalties: high sensitivity should pull the
        # *tier* down even if the complexity score is moderate.
        return _clip01(total)

    def route_from_score(self, scores: ParameterScores, weighted: float) -> str:
        if scores.latency_sensitivity >= 0.7 or scores.cost_sensitivity >= 0.7:
            if weighted < 0.55:
                return "nano"
        if weighted >= 0.58 or (
            scores.reasoning_complexity >= 0.65 and scores.accuracy_requirement >= 0.5
        ):
            return "ultra"
        if weighted >= 0.32:
            return "super"
        return "nano"

    def classify(self, prompt: str) -> ClassificationResult:
        scores = self.extractor.extract(prompt)
        weighted = round(self.weighted_score(scores), 4)
        tier = self.route_from_score(scores, weighted)
        return ClassificationResult(
            prompt=prompt,
            scores=scores,
            weighted_score=weighted,
            route_tier=tier,
            recommended_model=ROUTE_MODELS[tier],
            rationale={
                "top_drivers": self._top_drivers(scores),
                "task_type": scores.predicted_task_type,
            },
            source="heuristic",
        )

    def _top_drivers(self, scores: ParameterScores, k: int = 4) -> list[dict]:
        data = asdict(scores)
        contrib = []
        for name, weight in WEIGHTS.items():
            contrib.append(
                {
                    "parameter": name,
                    "score": data[name],
                    "weight": weight,
                    "contribution": round(weight * float(data[name]), 4),
                }
            )
        contrib.sort(key=lambda row: row["contribution"], reverse=True)
        return contrib[:k]


class PromptComplexityModel:
    """Random forest that maps the 14 scores to a routing tier."""

    FEATURES = list(WEIGHTS.keys())

    def __init__(self):
        self.extractor = PromptFeatureExtractor()
        self.heuristic = HeuristicRouter(self.extractor)
        self.model: RandomForestClassifier | None = None

    def _row(self, scores: ParameterScores) -> list[float]:
        data = asdict(scores)
        return [float(data[name]) for name in self.FEATURES]

    def generate_training_frame(self, n: int = 600, seed: int = 42) -> pd.DataFrame:
        rng = random.Random(seed)
        prompts = _synthetic_prompts(n, rng)
        rows = []
        for prompt, true_tier, true_task in prompts:
            scores = self.extractor.extract(prompt)
            row = asdict(scores)
            row["prompt"] = prompt
            row["label_tier"] = true_tier
            row["label_task"] = true_task
            rows.append(row)
        return pd.DataFrame(rows)

    def fit(self, df: pd.DataFrame | None = None, n: int = 600) -> dict:
        if df is None:
            df = self.generate_training_frame(n=n)
        x = df[self.FEATURES].to_numpy(dtype=float)
        y = df["label_tier"].to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42, stratify=y
        )
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred, labels=["nano", "super", "ultra"]).tolist()
        return {
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "accuracy": report["accuracy"],
            "classification_report": report,
            "confusion_matrix_labels": ["nano", "super", "ultra"],
            "confusion_matrix": matrix,
            "feature_importances": {
                name: float(imp)
                for name, imp in zip(self.FEATURES, self.model.feature_importances_)
            },
        }

    def save(self, path: Path = MODEL_PATH) -> Path:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self.model, fh)
        return path

    def load(self, path: Path = MODEL_PATH) -> None:
        with path.open("rb") as fh:
            self.model = pickle.load(fh)

    def classify(self, prompt: str, use_model: bool = True) -> ClassificationResult:
        heuristic_result = self.heuristic.classify(prompt)
        if not use_model or self.model is None:
            return heuristic_result
        scores = heuristic_result.scores
        x = np.array([self._row(scores)], dtype=float)
        tier = str(self.model.predict(x)[0])
        proba = {
            cls: float(p)
            for cls, p in zip(self.model.classes_, self.model.predict_proba(x)[0])
        }
        heuristic_result.route_tier = tier
        heuristic_result.recommended_model = ROUTE_MODELS[tier]
        heuristic_result.source = "random_forest"
        heuristic_result.rationale["class_probabilities"] = proba
        return heuristic_result


def _synthetic_prompts(n: int, rng: random.Random) -> list[tuple[str, str, str]]:
    """Diverse synthetic prompts with gold routing labels."""
    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "IWM", "NVDA"]
    strategies = ["SMA crossover", "RSI mean reversion", "volatility targeting", "Donchian breakout"]
    nano_fn = [
        lambda: (rng.choice(["hi", "thanks", "ok", "hello"]), "nano", "generation"),
        lambda: (f"what is an ETF? {rng.choice(['briefly', 'quickly', 'in one sentence'])}", "nano", "generation"),
        lambda: (f"define {rng.choice(['SMA', 'RSI', 'Sharpe', 'IV'])} quickly", "nano", "generation"),
        lambda: (
            f"classify this tweet as positive or negative: {rng.choice(['I love SPY', 'markets are ugly', 'meh'])}",
            "nano",
            "classification",
        ),
        lambda: (
            f"extract the ticker from: buy {rng.choice(tickers)} tomorrow",
            "nano",
            "extraction",
        ),
        lambda: (
            f"give a cheap short answer: what is a {rng.choice(['call', 'put', 'covered call'])} option?",
            "nano",
            "generation",
        ),
    ]
    super_fn = [
        lambda: (
            f"Write a {rng.randint(3, 6)}-paragraph explanation of {rng.choice(strategies)} "
            f"for a beginner using {rng.choice(tickers)}.",
            "super",
            "generation",
        ),
        lambda: (
            f"Analyze yesterday's {rng.choice(tickers)} daily bar and interpret whether "
            "trend or mean-reversion is more plausible.",
            "super",
            "analysis",
        ),
        lambda: (
            "Extract the following fields as JSON: ticker, side, size, and rationale from this journal note.",
            "super",
            "extraction",
        ),
        lambda: (
            f"Compare {rng.choice(strategies)} versus {rng.choice(strategies)}. "
            "List pros and cons, then choose one for a 1-week paper trial.",
            "super",
            "analysis",
        ),
        lambda: (
            "Draft a lesson plan covering options 101: call/put payoff and a covered call example with a table.",
            "super",
            "planning",
        ),
        lambda: (
            "Research common pitfalls when reading backtest Sharpe and max drawdown. Cite typical failure modes.",
            "super",
            "research",
        ),
        lambda: (
            "Implement a Python function that computes RSI(14) from a close-price list and return unit-testable code.",
            "super",
            "coding",
        ),
        lambda: (
            "Plan the remaining days of a 1-week quant curriculum with conditional branches if the user is advanced.",
            "super",
            "planning",
        ),
    ]
    ultra_fn = [
        lambda: (
            f"Optimize a long-only {rng.choice(tickers)} strategy under a {rng.choice(['8%', '10%', '12%'])} "
            "max-drawdown constraint. Compare at least three alternatives, debate trade-offs, "
            "derive position sizing from Kelly vs fixed-fractional, and output a structured JSON report "
            "plus Python backtest stub. Verify edge cases and do not hallucinate metrics.",
            "ultra",
            "planning",
        ),
        lambda: (
            "Derive the Black-Scholes call price step by step, show the PDE reduction, "
            "compute delta/gamma numerically, and validate against a closed form. "
            "Must be numerically precise. Then critique the lognormal assumption.",
            "ultra",
            "mathematical_reasoning",
        ),
        lambda: (
            "Design a production architecture for a tutor agent: memory recall, curriculum DAG, "
            "tool-restricted sandbox backtests, and an LLM router. Enumerate design alternatives, "
            "rank them, and produce a multi-section report. Constraints: no live brokerage, daily OHLC only.",
            "ultra",
            "planning",
        ),
        lambda: (
            "Write a robust Python backtesting module that must handle missing bars, "
            "corporate actions approximations, and walk-forward validation. Include tests, "
            "JSON metrics schema, and a self-check that Sharpe is not annualized incorrectly.",
            "ultra",
            "coding",
        ),
        lambda: (
            "Given 8 candidate option structures, search the space of defined-risk verticals "
            "versus covered calls. Optimize expected shortfall, then select one. "
            "Current implied vol and today's term structure are required. Use calculation tools.",
            "ultra",
            "analysis",
        ),
        lambda: (
            "Prove that a Kelly fraction f* = p/a - q/b under a binary payoff, then discuss estimation error.",
            "ultra",
            "mathematical_reasoning",
        ),
    ]

    mix = [("nano", nano_fn, 0.34), ("super", super_fn, 0.40), ("ultra", ultra_fn, 0.26)]
    out = []
    for _ in range(n):
        roll = rng.random()
        acc = 0.0
        fns = super_fn
        for _tier, group, p in mix:
            acc += p
            if roll <= acc:
                fns = group
                break
        prompt, tier, task = rng.choice(fns)()
        if rng.random() < 0.25:
            prompt += " " + rng.choice(
                ["Please be careful.", "Keep it concise.", "Show your reasoning.", "Must be correct."]
            )
        out.append((prompt, tier, task))
    return out


def classify_prompt(prompt: str, model_path: Path = MODEL_PATH) -> ClassificationResult:
    clf = PromptComplexityModel()
    if model_path.exists():
        clf.load(model_path)
        return clf.classify(prompt, use_model=True)
    return clf.classify(prompt, use_model=False)


def _print_result(result: ClassificationResult) -> None:
    print(json.dumps(result.to_dict(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a prompt for LLM routing.")
    parser.add_argument("--prompt", type=str, help="Prompt text to classify")
    parser.add_argument("--train", action="store_true", help="Train the random forest on synthetic data")
    parser.add_argument("--n", type=int, default=600, help="Synthetic sample count for --train")
    parser.add_argument("--save-data", type=str, default="", help="Optional CSV path for the training frame")
    args = parser.parse_args()

    clf = PromptComplexityModel()

    if args.train:
        metrics = clf.fit(n=args.n)
        path = clf.save()
        summary = {k: metrics[k] for k in ("n_train", "n_test", "accuracy", "feature_importances")}
        summary["saved_to"] = str(path)
        print(json.dumps(summary, indent=2))
        print("\nClassification report (test):")
        print(json.dumps(metrics["classification_report"], indent=2))
        if args.save_data:
            df = clf.generate_training_frame(n=args.n)
            Path(args.save_data).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save_data, index=False)
            print(f"Wrote {args.save_data}")
        return

    if not args.prompt:
        parser.error("pass --prompt '...' or --train")

    if MODEL_PATH.exists():
        clf.load(MODEL_PATH)
        result = clf.classify(args.prompt, use_model=True)
    else:
        result = clf.classify(args.prompt, use_model=False)
    _print_result(result)


if __name__ == "__main__":
    main()
