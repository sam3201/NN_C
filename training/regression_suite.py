from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from training.task_harness import TaskHarness
from training.teacher_pool import build_provider


def run_regression_suite(tasks_path: str, provider_spec: str, min_pass_rate: float,
                         max_examples: int | None = None, output_json: str | None = None,
                         timeout_s: int = 60) -> Dict[str, Any]:
    harness = TaskHarness()
    tasks = harness.load_tasks(tasks_path)
    provider = build_provider(provider_spec, timeout_s=timeout_s)
    result = harness.run_suite(tasks, provider, max_examples=max_examples)
    result["min_pass_rate"] = min_pass_rate
    result["passed_gate"] = result["pass_rate"] >= min_pass_rate
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression suite for growth safety")
    parser.add_argument("--tasks", required=True, help="Tasks JSONL path")
    parser.add_argument("--provider", required=True, help="Provider spec, e.g. ollama:mistral:latest")
    parser.add_argument("--min-pass", type=float, default=0.7)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--timeout", type=int, default=int(os.getenv("SAM_REGRESSION_TIMEOUT_S", "120")))
    args = parser.parse_args()

    result = run_regression_suite(
        tasks_path=args.tasks,
        provider_spec=args.provider,
        min_pass_rate=args.min_pass,
        max_examples=args.max_examples,
        output_json=args.output_json,
        timeout_s=args.timeout,
    )
    print(json.dumps(result, indent=2))
    if not result["passed_gate"]:
        raise SystemExit("Regression gate failed")


if __name__ == "__main__":
    main()
