import argparse
import json
import time
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP chatbot soak to grow groupchat distillation stream")
    parser.add_argument("--base-url", default="http://localhost:5004")
    parser.add_argument("--messages", type=int, default=20)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--distill-path", default="training/distilled/groupchat.jsonl")
    parser.add_argument("--prefix", default="Soak")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    distill_path = Path(args.distill_path)

    before_lines = 0
    if distill_path.exists():
        before_lines = sum(1 for _ in distill_path.open("r", encoding="utf-8"))

    for idx in range(args.messages):
        payload = {
            "message": f"{args.prefix} message {idx + 1}: confirm consensus + distillation.",
            "context": {"user_name": "Soak", "history": []},
        }
        resp = requests.post(f"{base_url}/api/chatbot", json=payload, timeout=60)
        if not resp.ok:
            print(f"[{idx + 1}] error: {resp.status_code} {resp.text[:120]}")
        else:
            data = resp.json()
            print(f"[{idx + 1}] ok: {data.get('response', '')[:80]}")
        time.sleep(args.delay)

    after_lines = before_lines
    latest = None
    if distill_path.exists():
        lines = distill_path.read_text(encoding="utf-8").strip().splitlines()
        after_lines = len(lines)
        if lines:
            latest = json.loads(lines[-1])

    print(json.dumps({
        "distill_path": str(distill_path),
        "lines_before": before_lines,
        "lines_after": after_lines,
        "lines_added": max(0, after_lines - before_lines),
        "latest_record": latest,
    }, indent=2))


if __name__ == "__main__":
    main()
