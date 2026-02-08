# SAM 2.0 AGI

SAM 2.0 is a hybrid Python/C multi-agent system with a web dashboard, slash-command interface, and C-accelerated cores for meta-control and dual-system simulation.

## What's Included
- Python orchestration, API server, and CLI: `complete_sam_unified.py`
- Web dashboard and terminal UI served by the API
- C extensions for speed: `sam_ananke_dual_system`, `sam_meta_controller_c`, `consciousness_*`, `multi_agent_orchestrator_c`, `specialized_agents_c`
- Support scripts and runners: `run_sam.sh`, `setup.py`

## Requirements
- Python 3.10+
- A C compiler toolchain compatible with Python extensions
- Optional local model backend: Ollama (if using local models)
- Optional hosted model backends: set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GITHUB_TOKEN`

## Quick Start
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Build C extensions
```bash
python setup.py build_ext --inplace
```
3. (Optional) Install Ollama and pull at least one model
```bash
ollama --version
ollama pull codellama:latest
```
4. Run
```bash
./run_sam.sh
# or
python3 complete_sam_unified.py
```

## Interfaces
- Dashboard: http://localhost:5004
- Terminal: http://localhost:5004/terminal
- Health/API: `/api/health`, `/api/agents`, `/api/command`, `/api/terminal/execute`

## Slash Commands (subset)
- `/help`, `/status`, `/agents`
- `/connect <agent_id>`, `/disconnect <agent_id>`, `/clone <agent_id> [name]`, `/spawn <type> <name> [personality]`
- `/research <topic>`, `/code <task>`, `/finance <query>`, `/websearch <query>`
- `/start`, `/stop`, `/clear`

## Dual System Implementation (SAM + ANANKE)
The C extension `sam_ananke_dual_system` implements a self-referential dual-system arena optimized for speed:
- Fast RNG (xorshift64*) and fixed-size arenas
- Internal state and long-term memory vectors per system
- Self-alignment and memory-energy metrics integrated into objective scoring
- Objective mutation with structural term changes and self-reference gain
- ANANKE kill confirmation term for adversarial termination pressure
- ANANKE unbounded mode (aggressive mutation + action scaling)
- Arena pressure feedback loop and adversarial interaction
- Python bindings for creation, stepping, mutation, and telemetry

## Meta-Controller (C)
The `sam_meta_controller_c` extension provides:
- Pressure aggregation across residuals, interference, retrieval entropy, and more
- Growth primitive selection (latent expansion, submodel spawn, routing, consolidation)
- Identity anchoring and invariant checks
- Objective contract evaluation (minimax-style)
- Policy gates: persistence thresholds, dominance margin, cooldowns, and risk caps

### Pressure Signals (SAM → Meta)
SAM emits only structured pressure channels:
- `residual`, `rank_def`, `retrieval_entropy`, `interference`
- `planner_friction`, `context_collapse`, `compression_waste`, `temporal_incoherence`

### Growth Primitives (Only Allowed Mutations)
- `GP_LATENT_EXPAND` (add latent dimensions)
- `GP_SUBMODEL_SPAWN` (split into specialized sub-models)
- `GP_INDEX_EXPAND` (expand memory index topology)
- `GP_ROUTING_INCREASE` (increase routing degree)
- `GP_CONTEXT_EXPAND` (expand context binding)
- `GP_PLANNER_WIDEN` (planner depth/width)
- `GP_CONSOLIDATE` (compression/pruning)
- `GP_REPARAM` (representation reparameterization)

### Invariants (Must Never Be Violated)
- Growth causality: every mutation must follow a valid pressure → selection → apply path
- Identity continuity: anchor similarity must remain above threshold
- Cooldown enforcement: structural changes are rate-limited
- Objective immutability (outside explicit contract evaluation)

## Repository Highlights
- `complete_sam_unified.py` — main orchestrator, API, and UI server
- `sam_ananke_dual_system.c` — dual-system arena
- `sam_meta_controller_c.c` — meta-controller core
- `multi_agent_orchestrator_c.c` — agent coordination
- `specialized_agents_c.c` — specialized agent primitives
- `consciousness_*.c` — consciousness-related modules

## Smoke Test
```bash
python3 -c "import sam_ananke_dual_system, sam_meta_controller_c; print('C extensions import OK')"
python3 -c "from complete_sam_unified import UnifiedSAMSystem; print('System import OK')"
```

## Comprehensive Tests
```bash
SAM_TEST_MODE=1 ./venv/bin/python -c "from SAM_AGI import CompleteSAMSystem; s=CompleteSAMSystem(); s.run_comprehensive_tests()"
```

## Training Pipeline
### 1) Install training requirements
```bash
pip install -r requirements_training.txt
```

### 2) Build a distillation dataset (teacher consensus)
```bash
python -m training.distillation \
  --tasks training/tasks/default_tasks.jsonl \
  --output training/distilled.jsonl \
  --teacher ollama:mistral:latest \
  --teacher ollama:llama3:latest \
  --n-per-teacher 1 \
  --min-similarity 0.72 \
  --min-votes 1
```

### 3) Train (LoRA or full fine‑tune)
```bash
# LoRA
python -m training.training_loop \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset training/distilled.jsonl \
  --output training/output_lora \
  --lora

# Full fine‑tune
python -m training.training_loop \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset training/distilled.jsonl \
  --output training/output_full \
  --full
```

### 4) Regression gate (blocks unsafe growth)
```bash
python -m training.regression_suite \
  --tasks training/tasks/default_tasks.jsonl \
  --provider ollama:mistral:latest \
  --min-pass 0.7
```

Environment overrides:
- `SAM_POLICY_PROVIDER` (e.g. `ollama:mistral:latest`)
- `SAM_REGRESSION_TASKS` (default: `training/tasks/default_tasks.jsonl`)
- `SAM_REGRESSION_MIN_PASS` (default: `0.7`)
- `SAM_REGRESSION_ON_GROWTH` (default: `1`)

## Failure Case Simulation
```bash
python3 ./simulate_failure_cases.py
```
