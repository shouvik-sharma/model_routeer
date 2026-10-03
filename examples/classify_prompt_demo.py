from tools.prompt_classifier import PromptComplexityModel, classify_prompt

SAMPLES = [
    "what is a put option? brief answer",
    "Compare SMA crossover vs RSI mean reversion for a 1-week SPY paper trial and pick one.",
    (
        "Optimize a long-only strategy under a 10% drawdown cap. Compare SMA, RSI, and vol targeting, "
        "derive sizing, output JSON plus Python, and verify edge cases. Do not hallucinate metrics."
    ),
]


def main() -> None:
    clf = PromptComplexityModel()
    print("Training routing model on synthetic labels...")
    metrics = clf.fit(n=500)
    clf.save()
    print(f"Test accuracy: {metrics['accuracy']:.3f}\n")

    for prompt in SAMPLES:
        result = classify_prompt(prompt)
        print("=" * 72)
        print(f"PROMPT: {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
        print(f"task_type: {result.scores.predicted_task_type}")
        print(f"weighted_score: {result.weighted_score:.3f}")
        print(f"route: {result.route_tier} -> {result.recommended_model}  [{result.source}]")
        print("top drivers:")
        for row in result.rationale["top_drivers"]:
            print(f"  {row['parameter']}: score={row['score']:.2f} contrib={row['contribution']:.3f}")


if __name__ == "__main__":
    main()
