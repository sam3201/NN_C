#!/usr/bin/env bash
set -euo pipefail

MODEL_LOCAL="${MODEL_LOCAL:-ollama/llama3}"

if [[ ! -f "ai/brief.md" ]]; then
  echo "Missing ai/brief.md. Run setup_free_agents.sh first."
  exit 1
fi

echo "==> 1) Generate a plan with Gemini CLI (optional)"
if command -v gemini >/dev/null; then
  gemini --sandbox -y -p "Read this brief and write a step-by-step implementation plan. Output markdown only." \
    < ai/brief.md > ai/plan.md || true
  echo "Wrote ai/plan.md"
else
  echo "Gemini CLI not found; skipping plan generation."
fi

echo "==> 2) Apply the plan with Aider (git-aware edits)"
# Edit the message to fit your task. You can also list files at the end for tighter scope.
INSTR="Implement the plan in ai/plan.md. If no plan exists, derive one from ai/brief.md. Make minimal, safe changes. Add/update tests. Explain how to run them."
aider --model "$MODEL_LOCAL" --message "$INSTR" .

echo "==> 3) Next: open VS Code + Continue to review diffs and iterate"
echo "Tip: keep prompts/decisions in ai/notes.md so every tool stays aligned."

