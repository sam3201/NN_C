# SAM-D Comprehensive Planning Guide
## Deep Scan Complete - 2026-02-13

---

## 1. CODEBASE INVENTORY

### 1.1 Directory Structure
```
NN_C/
├── src/
│   ├── python/          # 29 Python modules
│   │   ├── complete_sam_unified.py    # Main orchestrator (18K+ lines)
│   │   ├── sam_cores.py               # C integration + Phase 1-3 systems
│   │   ├── SAM_AGI.py                 # Legacy AGI wrapper
│   │   ├── survival_agent.py          # Survival metrics
│   │   ├── goal_management.py         # Goal tracking
│   │   ├── circuit_breaker.py         # Fault tolerance
│   │   ├── concurrent_executor.py     # Task execution
│   │   ├── vision_system.py           # Visual processing
│   │   ├── sensory_controller.py      # Input handling
│   │   ├── health_intelligence.py     # System health
│   │   ├── revenue_ops.py             # Revenue operations
│   │   ├── banking_ledger.py          # Financial tracking
│   │   ├── sam_code_modifier.py       # Code patching
│   │   ├── sam_code_scanner.py        # Code analysis
│   │   ├── sam_github_integration.py  # GitHub API
│   │   ├── sam_gmail_integration.py   # Gmail API
│   │   ├── sam_web_search.py          # Web research
│   │   ├── sam_auth_manager.py        # Authentication
│   │   ├── prompt_test_suite.py       # Prompt testing
│   │   └── [more modules...]
│   │
│   └── c_modules/       # 23 C source files
│       ├── sam_god_equation.c         # ΨΔ•Ω Core equation
│       ├── sam_regulator_compiler.c   # 53-regulator compiler
│       ├── sam_consciousness.c        # Consciousness module
│       ├── sam_memory.c               # Memory system
│       ├── sam_fast_rng.c             # Random number generator
│       ├── sam_telemetry_core.c       # Telemetry channels
│       ├── specialized_agents_c.c     # Agent implementations
│       ├── multi_agent_orchestrator_c.c
│       ├── consciousness_algorithmic.c
│       ├── sav_core_c.c               # SAV core
│       ├── sam_sav_dual_system.c      # Dual system
│       ├── sam_meta_controller_c.c    # Meta control
│       ├── sam_regulator_c.c          # Regulator
│       └── [legacy/deprecated files...]
│
├── include/             # 14 C header files
│
├── automation_framework/ # Hybrid Rust/Python
│   ├── src/            # Rust core (10 modules)
│   │   ├── lib.rs              # Main framework
│   │   ├── model_router.rs      # Dynamic model selection
│   │   ├── governance.rs        # Tri-cameral governance
│   │   ├── subagent.rs          # Subagent management
│   │   ├── resource.rs          # Resource tracking
│   │   ├── change_detection.rs  # Change tracking
│   │   ├── brittleness.rs       # Brittleness analysis
│   │   ├── completeness.rs     # Completeness checks
│   │   ├── workflow.rs          # Workflow execution
│   │   └── errors.rs            # Error handling
│   └── python/
│       ├── automation_bridge.py    # Python bindings
│       └── webhook_server.py        # Webhook communication
│
├── tools/              # Utility scripts
│   ├── experiment_framework.py      # NEW: Experiment runner
│   ├── run_recursive_checks.sh
│   ├── run_lora_gpu_7b.sh
│   └── [more...]
│
├── training/           # Training modules
│   ├── regression_suite.py
│   ├── teacher_pool.py
│   ├── training_loop.py
│   ├── distillation.py
│   └── task_harness.py
│
├── tests/              # Test suite (8 files)
│   ├── test_smoke.py
│   ├── test_orchestrator.py
│   ├── test_regression_gate.py
│   ├── test_governance_veto.py
│   └── [more...]
│
├── .opencode/          # OpenCode configuration
│   ├── opencode.json              # MCP servers + tools
│   └── tools/
│       ├── openclaw-tool.ts       # NEW: OpenClaw integration
│       ├── sam-tools.ts
│       └── subagent-orchestrator.ts
│
├── .openclaw/          # OpenClaw configuration
│   ├── config.json
│   ├── openclaw_bridge.py
│   └── master_integration.py
│
├── DOCS/               # 16 documentation files
│   ├── OMNISYNAPSE_X_COMPLETE_DOCUMENTATION.md
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── GOD_EQUATION.md
│   ├── ΨΔ•Ω-CORE_V5_IMPLEMENTATION_DETAILS.md
│   └── [more...]
│
├── skills/             # Agent skills
│   ├── anthropic/anthropic-reasoning/
│   └── sonoscli/
│
├── [Root files]
│   ├── setup.py                 # C extension build
│   ├── run_production.sh        # Production launcher
│   ├── run_sam.sh               # SAM launcher
│   ├── run_unified.sh           # Unified launcher
│   ├── AGENTS.md                # Coding guidelines
│   └── [config files...]
│
└── [Compiled C extensions - 18 .so files]
```

---

## 2. SYSTEM ARCHITECTURE

### 2.1 Core Components

#### ΨΔ•Ω-Core (God Equation)
- **Location**: `src/c_modules/sam_god_equation.c`
- **Purpose**: Master equation governing system evolution
- **State Variables**: K (knowledge), U (understanding), O (observations), ω (coherence)
- **Equation**: K' = K + (discovery - burden - contradiction_penalty) * dt
- **Python Binding**: `sam_god_equation` module

#### 53-Regulator System
- **Location**: `src/c_modules/sam_regulator_compiler.c`
- **Purpose**: Multi-dimensional constraint satisfaction
- **Groups**: 11 major groups (survival, coherence, growth, etc.)
- **Python Binding**: `sam_regulator_compiler_c` module

#### Phase 1: Core Psychology (sam_cores.py)
- **DriveSystem**: Id/Ego/Superego drives
- **EmotionSystem**: Valence, Arousal, Dominance
- **WisdomSystem**: Future-Preserving Coherence

#### Phase 2: Power & Control
- **PowerSystem**: Resources × Capabilities × Influence
- **ControlSystem**: Precision, Scope, Wisdom constraints
- **ResourceManager**: Compute, Memory, Energy, Budget

#### Phase 3: Meta Layer
- **MetaObserver**: Self-observation, introspection
- **CounterfactualEngine**: "What-if" reasoning
- **VersionalityTracker**: Evolution tracking

### 2.2 C Extensions (18 modules)
| Module | Purpose | Status |
|--------|---------|--------|
| sam_god_equation | ΨΔ•Ω Core | ✅ |
| sam_regulator_compiler | 53-Regulator | ✅ |
| sam_consciousness | Consciousness | ✅ |
| sam_memory | Memory system | ✅ |
| sam_fast_rng | RNG | ✅ |
| sam_telemetry_core | Telemetry (30 channels) | ✅ |
| specialized_agents_c | Agent implementations | ✅ |
| multi_agent_orchestrator_c | Orchestration | ✅ |
| consciousness_algorithmic | Algorithmic consciousness | ✅ |
| sav_core_c | SAV core | ✅ |
| sam_sav_dual_system | Dual system | ✅ |
| sam_meta_controller_c | Meta control | ✅ |
| sam_regulator_c | Regulator | ✅ |
| orchestrator_and_agents | Combined extension | ✅ |
| consciousness_algorithmic | Standalone | ✅ |

---

## 3. EXPERIMENT FRAMEWORK

### 3.1 Default Model Configuration
```python
# Kimi K2.5 - Best free model for experiments
DEFAULT_MODEL = "kimi:kimi-k2.5-flash"
FALLBACK_MODEL = "ollama:qwen2.5-coder:7b"
```

### 3.2 Experiment Categories
1. **C Extensions Check** - Verify all 12 modules build/import
2. **Python Syntax Check** - Verify 3 main files compile
3. **System Import Check** - Verify UnifiedSAMSystem imports
4. **API Providers Check** - Detect available models
5. **Fallback Patterns Check** - Find simulated/stub code
6. **Security Patterns Check** - Detect unsafe patterns

### 3.3 Running Experiments
```bash
# With Kimi (default)
KIMI_API_KEY=your_key python tools/experiment_framework.py

# With Ollama fallback
python tools/experiment_framework.py
```

---

## 4. INTEGRATIONS

### 4.1 OpenCode Integration
- **MCP Servers**: context7, github-search, filesystem, openclaw-webhook
- **Tools**: openclaw_execute, openclaw_tri_cameral, openclaw_status
- **Skills**: tri-cameral-orchestrator, cyclic-development-workflow

### 4.2 OpenClaw Integration
- **Webhook Server**: Port 8765
- **Endpoints**:
  - `/webhook` - General commands
  - `/webhook/execute` - Direct execution
  - `/webhook/tri-cameral` - Governance cycle
  - `/webhook/cycle` - Development cycle
  - `/health` - Health check
  - `/status` - System status

### 4.3 Kimi K2.5 Configuration
```bash
# Environment
SAM_USE_KIMI=1
KIMI_API_KEY=your_key
KIMI_ENDPOINT=https://api.moonshot.cn/v1
KIMI_MODEL=kimi-k2.5-flash
```

---

## 5. KNOWN ISSUES & GAPS

### 5.1 Critical Issues
| Issue | Location | Priority |
|-------|----------|----------|
| C extensions not importable in experiment context | experiment_framework.py | HIGH |
| 51 fallback patterns in code | complete_sam_unified.py | MEDIUM |
| 18 _fallback methods | various | MEDIUM |
| 10 simulated references | various | LOW |
| 4 stub implementations | various | LOW |

### 5.2 Missing Components
1. **Rust Core Compilation** - automation_framework/src/ is placeholder
2. **Kimi Integration** - API client not fully implemented
3. **Git Push** - samaisystemagi remote needs permissions

---

## 6. ACTION ITEMS

### 6.1 Immediate (Today)
- [ ] Rebuild C extensions: `python setup.py build_ext --inplace`
- [ ] Test experiment framework: `python tools/experiment_framework.py`
- [ ] Configure Kimi API key for experiments

### 6.2 Short-term (This Week)
- [ ] Replace fallback patterns with real implementations
- [ ] Complete Kimi integration in code
- [ ] Build Rust automation framework
- [ ] Test OpenCode-OpenClaw webhook communication

### 6.3 Medium-term (This Month)
- [ ] Implement all missing components
- [ ] Complete security audit
- [ ] Full system integration testing
- [ ] Deploy to production

---

## 7. COMMAND REFERENCE

### Build & Run
```bash
# Build C extensions
python setup.py build_ext --inplace

# Run production
./run_production.sh

# Run with Kimi
SAM_USE_KIMI=1 KIMI_API_KEY=xxx ./run_sam.sh
```

### Experiments
```bash
# Run all experiments
python tools/experiment_framework.py

# Run specific check
python -c "
from tools.experiment_framework import ExperimentFramework
f = ExperimentFramework()
r = f.check_c_extensions()
print(r)
"
```

### Webhooks
```bash
# Start webhook server
python automation_framework/python/webhook_server.py

# Test webhook
curl -X POST http://localhost:8765/webhook \
  -H "Content-Type: application/json" \
  -d '{"command": "build-extensions"}'
```

---

## 8. VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 5.2.0 | 2026-02-13 | OpenCode-OpenClaw integration, Kimi support, experiment framework |
| 5.1.0 | 2026-02-13 | Phase 3 complete (Meta Observer, Counterfactuals, Versionality) |
| 5.0.0 | 2026-02-13 | ΨΔ•Ω-Core recursive meta-evolution |
| 4.0.0 | Earlier | Multi-regulator system |

---

*Generated by Deep Scan - 2026-02-13*
