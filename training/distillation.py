from __future__ import annotations

import argparse
import json
from typing import Dict, List

from training.task_harness import TaskHarness
from training.teacher_pool import TeacherPool, build_provider


def build_dataset(tasks_path: str, output_path: str, teacher_specs: List[str], n_per_teacher: int,
                  min_similarity: float, min_votes: int) -> Dict[str, int]:
    harness = TaskHarness()
    tasks = harness.load_tasks(tasks_path)
    providers = [build_provider(spec) for spec in teacher_specs]
    pool = TeacherPool(providers, min_similarity=min_similarity, min_votes=min_votes)

    kept = 0
    total = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for task in tasks:
            total += 1
            candidates = pool.generate(task.prompt, n_per_teacher=n_per_teacher)
            consensus = pool.consensus_filter(candidates)
            for cand in consensus:
                result = harness.score(task, cand.response)
                record = {
                    "task_id": task.id,
                    "prompt": task.prompt,
                    "response": cand.response,
                    "score": result.score,
                    "passed": result.passed,
                    "scorer": task.scorer,
                    "teacher": {"provider": cand.provider, "model": cand.model, "latency_s": cand.latency_s},
                    "metadata": task.metadata,
                }
                out.write(json.dumps(record) + "\n")
                kept += 1
    return {"tasks": total, "records": kept}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build distillation dataset with teacher consensus filtering")
    parser.add_argument("--tasks", required=True, help="Path to tasks JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--teacher", action="append", required=True, help="Teacher spec, e.g. ollama:mistral:latest")
    parser.add_argument("--n-per-teacher", type=int, default=1)
    parser.add_argument("--min-similarity", type=float, default=0.72)
    parser.add_argument("--min-votes", type=int, default=1)
    args = parser.parse_args()

    stats = build_dataset(
        tasks_path=args.tasks,
        output_path=args.output,
        teacher_specs=args.teacher,
        n_per_teacher=args.n_per_teacher,
        min_similarity=args.min_similarity,
        min_votes=args.min_votes,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
