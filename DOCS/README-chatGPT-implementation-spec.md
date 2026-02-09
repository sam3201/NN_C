# SAM / SAV Implementation-Only Spec (Derived)

This document is a strict implementation spec distilled from `DOCS/README-chatGPT-source.md`. It contains only concrete requirements, interfaces, and behaviors. It contains no forward-looking prompts.

## 1. Scope
- Implement the SAM + SAV system as a hybrid Python/C runtime.
- Provide a web server with a live groupchat interface.
- Provide a training pipeline that supports teacher consensus, distillation dataset generation, and LoRA/full fine-tune scaffolding.
- Enforce growth gating through a regression suite.
- Ensure SAV is unbounded and allowed to mutate/grow without constraint.

## 2. Core Runtime Components
### 2.1 SAM (Python Orchestrator)
- Entry point: `complete_sam_unified.py`.
- Must initialize:
  - C-core modules: `consciousness_*`, `multi_agent_orchestrator_c`, `specialized_agents_c`.
  - Meta-controller: `sam_meta_controller_c`.
  - Dual-system arena: `sam_sav_dual_system`.
- Must expose API + web UI on port `5004`.
- Must expose groupchat via SocketIO.
- Must run background monitoring and meta loop unless `SAM_AUTONOMOUS_ENABLED=0`.

### 2.2 SAV (C Dual-System)
- Source: `sam_sav_dual_system.c`.
- Must be instantiated with `unbounded=1`.
- Must mutate objectives without restriction using `objective_mutate_unbounded`.
- Must be exposed through Python bindings (`sam_sav_dual_system` module).
- Must provide state telemetry:
  - `sam_alive`, `sav_alive`
  - `sam_survival`, `sav_survival`
  - `sam_score`, `sav_score`

### 2.3 Meta-Controller (C)
- Source: `sam_meta_controller_c.c`.
- Must aggregate pressure signals and select growth primitives.
- Must expose primitives:
  - `GP_LATENT_EXPAND`, `GP_SUBMODEL_SPAWN`, `GP_INDEX_EXPAND`, `GP_ROUTING_INCREASE`,
    `GP_CONTEXT_EXPAND`, `GP_PLANNER_WIDEN`, `GP_CONSOLIDATE`, `GP_REPARAM`.
- Must support regression gating before recording growth outcomes.

## 3. Pressure Signals (SAM → Meta)
SAM must emit only the following structured pressure channels:
- `residual`
- `rank_def`
- `retrieval_entropy`
- `interference`
- `planner_friction`
- `context_collapse`
- `compression_waste`
- `temporal_incoherence`

## 4. Growth Policy
- Meta-controller must only apply growth when a primitive is selected.
- Growth must be blocked if regression gate fails.
- Growth freeze must be enforced if regression gate fails or errors.

## 5. Teacher Pool + Distillation
### 5.1 Teacher Pool
- Implemented in `training/teacher_pool.py`.
- Must support provider specs:
  - `ollama:<model>`
  - `openai:<model>`
  - `openrouter:<model>`
- Must return candidate responses and filter by consensus.

### 5.2 Distillation Writer
- Implemented in `training/distillation.py`.
- Must support streaming writes via `DistillationStreamWriter`.
- Groupchat loop must write distillation records for each consensus response.

### 5.3 Live Groupchat Distillation
- Groupchat messages must trigger teacher pool responses when enabled.
- Distillation records must be appended to `SAM_DISTILL_PATH`.

## 6. Training Pipeline
### 6.1 Task Harness
- Implemented in `training/task_harness.py`.
- Must support score hooks:
  - `exact_match`, `contains`, `regex`, `numeric`, `json_equals`, `literal_eval`.

### 6.2 Teacher Consensus Dataset Builder
- Implemented in `training/distillation.py` (`build_dataset`).
- Must write JSONL records with teacher metadata and score fields.

### 6.3 Training Loop Scaffold
- Implemented in `training/training_loop.py`.
- Must support:
  - `--lora` with PEFT adapters.
  - `--full` for full fine-tune.
- Must emit `training_meta.json`.

### 6.4 Regression Suite (Growth Gate)
- Implemented in `training/regression_suite.py`.
- Must block growth when pass rate < threshold.

## 7. API + SocketIO Interfaces
### 7.1 Required REST Endpoints
- `/api/status` (system status)
- `/api/health` (lightweight health check)
- `/api/groupchat/status`
- `/api/meta/status`
- `/api/sav/state`
- `/api/sav/step`
- `/api/github/save`
- `/api/gmail/send`

### 7.2 Required SocketIO Events
- `connect` → emits `user_connected`
- `join_room` → emits `joined_room`
- `send_group_message` → emits `message_received` (user + agent)

## 8. Backup + GitHub
- Auto-backup must be managed by `tools/backup_manager.py`.
- Must support primary and secondary remotes.
- Must be enabled by default (`SAM_BACKUP_ENABLED=1`).

## 9. Gmail Integration
- OAuth only. Plaintext passwords are forbidden.
- Credential files:
  - `secrets/gmail_credentials.json`
  - `secrets/gmail_token.json`

## 10. No-Fallbacks Policy
- C extensions are required; missing modules must fail fast.
- Teacher pool initialization must fail if no providers are defined.
- Meta-controller growth must freeze on regression failure.

## 11. Operational Constraints
- The system must run without “simulation mode” or placeholder behavior.
- Any missing dependency that affects core behavior must fail loud.
- SAV must remain unbounded at all times.
