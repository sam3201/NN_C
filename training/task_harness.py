from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class Task:
    id: str
    prompt: str
    expected: Optional[Any] = None
    scorer: str = "exact_match"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    response: str
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


class ScorerRegistry:
    def __init__(self) -> None:
        self._scorers: Dict[str, Callable[[Task, str], Tuple[float, Dict[str, Any]]]] = {}
        self._register_defaults()

    def register(self, name: str, fn: Callable[[Task, str], Tuple[float, Dict[str, Any]]]) -> None:
        self._scorers[name] = fn

    def score(self, task: Task, response: str) -> Tuple[float, Dict[str, Any]]:
        fn = self._scorers.get(task.scorer)
        if not fn:
            raise ValueError(f"Unknown scorer: {task.scorer}")
        return fn(task, response)

    def _register_defaults(self) -> None:
        self.register("exact_match", self._score_exact)
        self.register("contains", self._score_contains)
        self.register("regex", self._score_regex)
        self.register("numeric", self._score_numeric)
        self.register("json_equals", self._score_json)
        self.register("literal_eval", self._score_literal_eval)

    @staticmethod
    def _normalize(text: str) -> str:
        return text.strip()

    def _score_exact(self, task: Task, response: str) -> Tuple[float, Dict[str, Any]]:
        expected = "" if task.expected is None else str(task.expected)
        got = self._normalize(response)
        exp = self._normalize(expected)
        score = 1.0 if got == exp else 0.0
        return score, {"expected": exp, "got": got}

    def _score_contains(self, task: Task, response: str) -> Tuple[float, Dict[str, Any]]:
        expected = "" if task.expected is None else str(task.expected)
        got = response
        score = 1.0 if expected in got else 0.0
        return score, {"expected": expected, "got": got}

    def _score_regex(self, task: Task, response: str) -> Tuple[float, Dict[str, Any]]:
        pattern = "" if task.expected is None else str(task.expected)
        ok = re.search(pattern, response, re.MULTILINE) is not None
        return (1.0 if ok else 0.0), {"pattern": pattern}

    def _score_numeric(self, task: Task, response: str) -> Tuple[float, Dict[str, Any]]:
        expected = float(task.expected)
        tol = float(task.metadata.get("tolerance", 1e-3))
        numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
        got = float(numbers[0]) if numbers else float("nan")
        score = 1.0 if abs(got - expected) <= tol else 0.0
        return score, {"expected": expected, "got": got, "tolerance": tol}

    def _score_json(self, task: Task, response: str) -> Tuple[float, Dict[str, Any]]:
        expected = task.expected
        try:
            got = json.loads(response)
        except Exception:
            return 0.0, {"error": "invalid_json"}
        score = 1.0 if got == expected else 0.0
        return score, {"expected": expected, "got": got}

    def _score_literal_eval(self, task: Task, response: str) -> Tuple[float, Dict[str, Any]]:
        expected = task.expected
        try:
            got = ast.literal_eval(response.strip())
        except Exception:
            return 0.0, {"error": "literal_eval_failed"}
        score = 1.0 if got == expected else 0.0
        return score, {"expected": expected, "got": got}


class TaskHarness:
    def __init__(self, scorer_registry: Optional[ScorerRegistry] = None) -> None:
        self.scorers = scorer_registry or ScorerRegistry()

    def load_tasks(self, path: str) -> List[Task]:
        tasks: List[Task] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                tasks.append(Task(
                    id=str(payload["id"]),
                    prompt=payload["prompt"],
                    expected=payload.get("expected"),
                    scorer=payload.get("scorer", "exact_match"),
                    metadata=payload.get("metadata", {}),
                ))
        return tasks

    def score(self, task: Task, response: str) -> TaskResult:
        score, details = self.scorers.score(task, response)
        return TaskResult(
            task_id=task.id,
            response=response,
            score=score,
            passed=score >= 0.5,
            details=details,
        )

    def run_task(self, task: Task, provider) -> TaskResult:
        response = provider.generate(task.prompt)
        if isinstance(response, tuple):
            response = response[0]
        return self.score(task, response)

    def run_suite(self, tasks: Iterable[Task], provider, max_examples: Optional[int] = None) -> Dict[str, Any]:
        results: List[TaskResult] = []
        for idx, task in enumerate(tasks):
            if max_examples is not None and idx >= max_examples:
                break
            results.append(self.run_task(task, provider))
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg = sum(r.score for r in results) / total if total else 0.0
        return {
            "total": total,
            "passed": passed,
            "pass_rate": (passed / total) if total else 0.0,
            "avg_score": avg,
            "results": [r.__dict__ for r in results],
        }
