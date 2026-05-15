from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from lab7_evaluation import RESULTS_PATH, REPORT_PATH, run_evaluation, write_report, write_results

THRESHOLD_CONFIG_PATH = Path(os.getenv("EVAL_THRESHOLD_CONFIG_PATH", "eval_thresholds.json"))
SUMMARY_RESULTS_PATH = Path(os.getenv("EVAL_SUMMARY_RESULTS_PATH", "ci_eval_results.json"))


def load_thresholds() -> dict[str, float]:
    return json.loads(THRESHOLD_CONFIG_PATH.read_text(encoding="utf-8"))


def summarize_metrics() -> dict[str, float]:
    runs = run_evaluation()
    write_results(runs)
    write_report(runs)

    return {
        "average_faithfulness": sum(run.faithfulness for run in runs) / len(runs),
        "average_relevancy": sum(run.answer_relevancy for run in runs) / len(runs),
        "average_tool_call_accuracy": sum(run.tool_call_accuracy for run in runs) / len(runs),
    }


def write_summary(metrics: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    metric_rows = [
        {
            "metric": "faithfulness",
            "score": round(metrics["average_faithfulness"], 3),
            "threshold": thresholds["min_faithfulness"],
            "passed": metrics["average_faithfulness"] >= thresholds["min_faithfulness"],
        },
        {
            "metric": "relevancy",
            "score": round(metrics["average_relevancy"], 3),
            "threshold": thresholds["min_relevancy"],
            "passed": metrics["average_relevancy"] >= thresholds["min_relevancy"],
        },
        {
            "metric": "tool_call_accuracy",
            "score": round(metrics["average_tool_call_accuracy"], 3),
            "threshold": thresholds["min_tool_call_accuracy"],
            "passed": metrics["average_tool_call_accuracy"] >= thresholds["min_tool_call_accuracy"],
        },
    ]
    failures = [row["metric"] for row in metric_rows if not row["passed"]]
    payload = {
        "dataset": os.getenv("TEST_DATASET_PATH", "test_dataset.json"),
        "threshold_config": str(THRESHOLD_CONFIG_PATH),
        "break_mode": os.getenv("BREAK_AGENT_FOR_CI", "false").lower() == "true",
        "overall_passed": not failures,
        "failed_metrics": failures,
        "metrics": metric_rows,
        "case_results_path": str(RESULTS_PATH),
        "report_path": str(REPORT_PATH),
    }
    SUMMARY_RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return failures


def main() -> int:
    thresholds = load_thresholds()
    metrics = summarize_metrics()
    failures = write_summary(metrics, thresholds)

    print("CI Evaluation Summary")
    print("=====================")
    print(f"Dataset: {os.getenv('TEST_DATASET_PATH', 'test_dataset.json')}")
    print(f"Threshold config: {THRESHOLD_CONFIG_PATH}")
    print(f"Results file: {RESULTS_PATH}")
    print(f"Report file: {REPORT_PATH}")
    print(f"Summary file: {SUMMARY_RESULTS_PATH}")
    if os.getenv("BREAK_AGENT_FOR_CI", "false").lower() == "true":
        print("Break mode: enabled (runtime plan intentionally degraded)")
    else:
        print("Break mode: disabled")

    print(f"Average faithfulness: {metrics['average_faithfulness']:.3f}")
    print(f"Average relevancy: {metrics['average_relevancy']:.3f}")
    print(f"Average tool accuracy: {metrics['average_tool_call_accuracy']:.3f}")

    if failures:
        print(f"Quality gate failed: {', '.join(failures)} below threshold.")
        return 1

    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
