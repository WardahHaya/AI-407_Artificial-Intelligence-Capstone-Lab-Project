from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from lab7_evaluation import RESULTS_PATH, REPORT_PATH, run_evaluation, write_report, write_results

THRESHOLD_CONFIG_PATH = Path(os.getenv("EVAL_THRESHOLD_CONFIG_PATH", "eval_threshold_config.json"))


def load_thresholds() -> dict[str, float]:
    return json.loads(THRESHOLD_CONFIG_PATH.read_text(encoding="utf-8"))


def maybe_apply_breaking_change(metrics: dict[str, float]) -> dict[str, float]:
    if os.getenv("BREAK_AGENT_FOR_CI", "false").lower() != "true":
        return metrics

    degraded = dict(metrics)
    degraded["average_faithfulness"] = round(metrics["average_faithfulness"] * 0.45, 3)
    degraded["average_relevancy"] = round(metrics["average_relevancy"] * 0.55, 3)
    degraded["average_tool_call_accuracy"] = round(metrics["average_tool_call_accuracy"] * 0.7, 3)
    return degraded


def summarize_metrics() -> dict[str, float]:
    runs = run_evaluation()
    write_results(runs)
    write_report(runs)

    metrics = {
        "average_faithfulness": sum(run.faithfulness for run in runs) / len(runs),
        "average_relevancy": sum(run.answer_relevancy for run in runs) / len(runs),
        "average_tool_call_accuracy": sum(run.tool_call_accuracy for run in runs) / len(runs),
    }
    return maybe_apply_breaking_change(metrics)


def main() -> int:
    thresholds = load_thresholds()
    metrics = summarize_metrics()

    print("CI Evaluation Summary")
    print("=====================")
    print(f"Dataset: {os.getenv('TEST_DATASET_PATH', 'test_dataset.json')}")
    print(f"Threshold config: {THRESHOLD_CONFIG_PATH}")
    print(f"Results file: {RESULTS_PATH}")
    print(f"Report file: {REPORT_PATH}")
    if os.getenv("BREAK_AGENT_FOR_CI", "false").lower() == "true":
        print("Break mode: enabled (simulated grounding/prompt failure)")
    else:
        print("Break mode: disabled")

    print(f"Average faithfulness: {metrics['average_faithfulness']:.3f}")
    print(f"Average relevancy: {metrics['average_relevancy']:.3f}")
    print(f"Average tool accuracy: {metrics['average_tool_call_accuracy']:.3f}")

    failures: list[str] = []
    if metrics["average_faithfulness"] < thresholds["min_faithfulness"]:
        failures.append("faithfulness")
    if metrics["average_relevancy"] < thresholds["min_relevancy"]:
        failures.append("relevancy")
    if metrics["average_tool_call_accuracy"] < thresholds["min_tool_call_accuracy"]:
        failures.append("tool_call_accuracy")

    if failures:
        print(f"Quality gate failed: {', '.join(failures)} below threshold.")
        return 1

    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
