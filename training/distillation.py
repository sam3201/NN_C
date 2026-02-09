from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from training.task_harness import TaskHarness
from training.teacher_pool import TeacherPool, build_provider


class DistillationStreamWriter:
    """Append-only writer for streaming distillation records."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record)
        with self._lock:
            with open(self.output_path, "a", encoding="utf-8") as out:
                out.write(line + "\n")


def _setup_logger(level: str, log_file: str | None) -> logging.Logger:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("distillation")


def build_dataset(
    tasks_path: str,
    output_path: str,
    teacher_specs: List[str],
    n_per_teacher: int,
    min_similarity: float,
    min_votes: int,
    max_tokens: int,
    timeout_s: int,
    log_every: int,
    logger: logging.Logger,
) -> Dict[str, int]:
    harness = TaskHarness()
    tasks = harness.load_tasks(tasks_path)
    providers = [build_provider(spec, max_tokens=max_tokens, timeout_s=timeout_s) for spec in teacher_specs]
    pool = TeacherPool(providers, min_similarity=min_similarity, min_votes=min_votes)

    logger.info("Loaded %d tasks from %s", len(tasks), tasks_path)
    logger.info("Teachers: %s", ", ".join(teacher_specs))
    logger.info("n_per_teacher=%d max_tokens=%d timeout_s=%d min_similarity=%.2f min_votes=%d",
                n_per_teacher, max_tokens, timeout_s, min_similarity, min_votes)

    kept = 0
    total = 0
    start_all = time.time()
    with open(output_path, "w", encoding="utf-8") as out:
        for idx, task in enumerate(tasks, start=1):
            total += 1
            task_start = time.time()
            logger.debug("Task %s prompt_len=%d", task.id, len(task.prompt))
            candidates = pool.generate(task.prompt, n_per_teacher=n_per_teacher)
            consensus = pool.consensus_filter(candidates)
            logger.debug("Task %s candidates=%d consensus=%d", task.id, len(candidates), len(consensus))
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
            if log_every and idx % log_every == 0:
                elapsed = time.time() - start_all
                logger.info("Progress %d/%d | kept=%d | last_task_s=%.2f | total_s=%.1f",
                            idx, len(tasks), kept, time.time() - task_start, elapsed)
    return {"tasks": total, "records": kept}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build distillation dataset with teacher consensus filtering")
    parser.add_argument("--tasks", required=True, help="Path to tasks JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--teacher", action="append", required=True, help="Teacher spec, e.g. ollama:mistral:latest")
    parser.add_argument("--n-per-teacher", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("SAM_TEACHER_MAX_TOKENS", "256")))
    parser.add_argument("--timeout-s", type=int, default=int(os.getenv("SAM_TEACHER_TIMEOUT_S", "180")))
    parser.add_argument("--log-level", default=os.getenv("SAM_TRAIN_LOG_LEVEL", "INFO"))
    parser.add_argument("--log-file", default=os.getenv("SAM_TRAIN_LOG_FILE"))
    parser.add_argument("--log-every", type=int, default=int(os.getenv("SAM_TRAIN_LOG_EVERY", "10")))
    parser.add_argument("--min-similarity", type=float, default=0.72)
    parser.add_argument("--min-votes", type=int, default=1)
    args = parser.parse_args()

    logger = _setup_logger(args.log_level, args.log_file)
    stats = build_dataset(
        tasks_path=args.tasks,
        output_path=args.output,
        teacher_specs=args.teacher,
        n_per_teacher=args.n_per_teacher,
        min_similarity=args.min_similarity,
        min_votes=args.min_votes,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        log_every=args.log_every,
        logger=logger,
    )
    logger.info("Completed distillation: %s", json.dumps(stats))


if __name__ == "__main__":
    main()
