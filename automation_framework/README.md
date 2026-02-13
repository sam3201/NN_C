# Automation Framework - COMPLETE SYSTEM

## 🎉 Status: FULLY OPERATIONAL

A comprehensive, standalone automation framework with **dynamic model routing**, **tri-cameral governance**, and **concurrent subagents** - completely separate from SAM-D (modular architecture).

---

## 📦 What's Been Built

### 1. 🦀 Rust Core (High Performance + Security)

**Location**: `automation_framework/src/`

**Modules**:
- ✅ `lib.rs` - Main framework interface
- ✅ `subagent.rs` - Concurrent subagent pool (semaphore-based)
- ✅ `governance.rs` - CIC/AEE/CSF tri-cameral system
- ✅ `resource.rs` - Billing & quota management
- ✅ `workflow.rs` - Cyclic development workflows
- ✅ `model_router.rs` - **Dynamic model selection**
- ✅ `change_detection.rs` - Smart change tracking
- ✅ `brittleness.rs` - Race condition detection
- ✅ `completeness.rs` - Verification system
- ✅ `errors.rs` - Comprehensive error handling

**Key Features**:
- Tokio async runtime for concurrency
- DashMap for lock-free concurrent data structures
- Rayon for data parallelism
- Parking lot for fast synchronization
- Memory-safe, thread-safe by design

### 2. 🐍 Python Bridge (Flexibility)

**Location**: `automation_framework/python/automation_bridge.py`

**Components**:
- ✅ `TriCameralOrchestrator` - Governance implementation
- ✅ `SubagentPool` - Parallel task execution
- ✅ `CyclicWorkflow` - Workflow management
- ✅ **NEW: `ModelRouter`** - Dynamic model selection
- ✅ `AutomationFramework` - Main interface

### 3. 🎯 Dynamic Model Router (Star Feature)

**Automatically selects best AI model for each task**:

```python
# Simple usage
model = select_best_model("Implement secure authentication")
# → 'claude-3-5-sonnet' (high reliability)

# With auto-switching
model = auto_switch_model("Quick code review")
# → 'claude-3-haiku' (fast & cheap)

# Full framework integration
result = await framework.execute_workflow(config)
# Automatically selects optimal model
```

**Smart Analysis**:
- Detects task type (coding, analysis, reasoning, creative)
- Assesses complexity (0.0-1.0)
- Identifies safety-critical tasks
- Considers context size
- Tracks time sensitivity

**Scoring Algorithm**:
```
Score = 
  Capability Match × 40% +
  Specialty Match × 20% +
  Context Fit × 15% +
  Cost Optimization × 15% +
  Historical Performance × 10%
```

**Budget-Aware**:
- Low usage (< 30%): Prioritize quality
- Medium (30-70%): Balance cost/quality
- High (> 70%): Prioritize cost

**Registered Models**:
- **Claude 3.5 Sonnet**: Best for reasoning, coding, safety
- **Claude 3 Haiku**: Fast, cheap for quick tasks
- **GPT-4**: High creativity and reasoning
- **Local LLM**: Zero cost, moderate quality

---

## 🏛️ Tri-Cameral Governance

### Three Branches

**CIC (Constructive Intelligence Core)**
- Role: Builder, Optimistic
- Actions: Plan, implement, optimize
- Vote: "Proceed with growth!"

**AEE (Adversarial Exploration Engine)**
- Role: Critic, Pessimistic
- Actions: Challenge, find edge cases
- Vote: "This will fail because..."

**CSF (Coherence Stabilization Field)**
- Role: Guardian, Neutral
- Actions: Validate invariants
- Vote: "Invariant check: PASS/FAIL"

### Decision Matrix

| CIC | AEE | CSF | Decision | Action |
|-----|-----|-----|----------|--------|
| YES | YES | YES | ✅ PROCEED | Execute |
| YES | NO | YES | ⚠️ REVISE | Address concerns |
| YES | YES | NO | 🛑 REJECT | Violates invariants |

---

## 🔄 Cyclic Workflow

```
START
  ↓
┌──────────┐
│   PLAN   │◄──────────────────┐
└────┬─────┘                    │
     ↓                          │
┌──────────┐  NO  ┌─────────┐  │
│ ANALYZE  │─────▶│ REVISE  │──┘
└────┬─────┘      └─────────┘
     │ YES
     ↓
┌──────────┐
│  BUILD   │
└────┬─────┘
     ↓
┌──────────┐  NO  ┌──────────┐
│ ANALYZE  │─────▶│ REFACTOR │────┐
└────┬─────┘      └──────────┘    │
     │ YES                         │
     ↓                             │
┌──────────┐                      │
│   TEST   │──────────────────────┘
└────┬─────┘
     ↓
┌──────────┐  NO  ┌────────┐
│ ANALYZE  │─────▶│ DEBUG  │────┐
└────┬─────┘      └────────┘    │
     │ YES                      │
     ↓                          │
┌──────────┐                   │
│ COMPLETE │◀───────────────────┘
└──────────┘
```

---

## 🚀 Concurrent Subagents

### Pattern 1: Parallel Execution
```python
results = framework.spawn_subagents(
    tasks=[task1, task2, task3],
    handler=process_task
)
# All 3 run simultaneously
```

### Pattern 2: Pipeline (Reader → Processor → Writer)
```python
result = subagent_pool.spawn_pipeline(
    task=data,
    reader=read_chunk,
    processor=analyze,
    writer=write_results
)
```

### Pattern 3: Verification (Multiple Checkers)
```python
results = subagent_pool.spawn_verifiers(
    task=code,
    verifiers=[check_syntax, check_security, check_performance]
)
```

**Concurrency Control**:
- Semaphore-based (max 10 concurrent)
- Priority levels (Low/Normal/High/Critical)
- Timeout support
- Retry logic
- Metrics tracking

---

## 💰 Resource Management

### Billing & Quotas

**Tracked Resources**:
- API calls per minute
- Tokens per hour
- Compute seconds per day
- Storage MB
- Daily budget ($100 default)

**Automatic Alerts**:
- 50% usage: Info notice
- 75% usage: Warning
- 90% usage: Critical alert

**Cost Optimization**:
- Dynamic budget tier adjustment
- Priority-based allocation
- Automatic cost tracking
- Hourly usage breakdown

---

## 🔧 Usage Examples

### Example 1: Basic Workflow
```python
from automation_bridge import AutomationFramework, WorkflowConfig

async def main():
    framework = AutomationFramework()
    
    config = WorkflowConfig(
        name="Implement Phase 2",
        high_level_plan="Add Power/Control systems",
        low_level_plan="Create P_t and C_t classes",
        invariants=["maintain_API_compat"],
        risk_level=0.7
    )
    
    result = await framework.execute_workflow(config)
    print(f"Model used: {result['model_used']}")
    print(f"Success: {result['success']}")

asyncio.run(main())
```

### Example 2: Dynamic Model Switching
```python
from automation_bridge import auto_switch_model

# Each task gets optimal model
tasks = [
    ("Quick check", "claude-3-haiku"),  # Fast
    ("Deep reasoning", "claude-3-5-sonnet"),  # Powerful
    ("Security audit", "claude-3-5-sonnet"),  # Reliable
]

for task, expected in tasks:
    model = auto_switch_model(task)
    print(f"{task}: {model}")
```

### Example 3: Parallel Subagents
```python
from automation_bridge import spawn_parallel_subagents

tasks = [
    {"file": "src/main.py", "action": "analyze"},
    {"file": "src/lib.rs", "action": "analyze"},
    {"file": "src/utils.py", "action": "analyze"},
]

def analyze_file(task):
    # Do analysis
    return f"Analyzed {task['file']}"

results = spawn_parallel_subagents(tasks, analyze_file)
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              AUTOMATION FRAMEWORK (Standalone)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              DYNAMIC MODEL ROUTER                     │  │
│  │  • Task analysis  • Model scoring  • Auto-switching  │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────┐  │
│  │   CIC        │     │    AEE       │     │    CSF     │  │
│  │ (Builder)    │     │   (Critic)   │     │ (Guardian) │  │
│  └──────┬───────┘     └──────┬───────┘     └─────┬──────┘  │
│         │                    │                   │          │
│         └────────────────────┼───────────────────┘          │
│                              ▼                              │
│                    ┌─────────────────┐                      │
│                    │ Decision Matrix │                      │
│                    └────────┬────────┘                      │
│                             │                               │
│         ┌───────────────────┼───────────────────┐          │
│         ▼                   ▼                   ▼          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │   PLAN       │──▶│   BUILD      │──▶│    TEST      │  │
│  └──────────────┘   └──────────────┘   └──────────────┘  │
│         │                   │                   │          │
│         └───────────────────┼───────────────────┘          │
│                             ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            CONCURRENT SUBAGENT POOL                   │  │
│  │  • Parallel execution  • Pipelines  • Verifiers      │  │
│  └──────────────────────────────────────────────────────┘  │
│                             │                               │
│                             ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              RESOURCE MANAGEMENT                      │  │
│  │  • Budget tracking  • Quotas  • Billing alerts       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         │                                    │
         ▼                                    ▼
┌──────────────────┐              ┌──────────────────┐
│   SAM-D System   │              │   Other Systems  │
│  (Being Built)   │              │  (Can use this)  │
└──────────────────┘              └──────────────────┘
```

**Key**: This automation framework is **completely modular** and **standalone**. It can orchestrate the building of SAM-D or any other system.

---

## 📁 File Structure

```
automation_framework/
├── Cargo.toml                  # Rust configuration
├── src/
│   ├── lib.rs                  # Main library
│   ├── subagent.rs             # Concurrent subagents
│   ├── governance.rs           # Tri-cameral governance
│   ├── resource.rs             # Billing & quotas
│   ├── model_router.rs         # Dynamic model selection ⭐
│   ├── workflow.rs             # Cyclic workflows
│   ├── change_detection.rs     # Smart change tracking
│   ├── brittleness.rs          # Race detection
│   ├── completeness.rs         # Verification
│   └── errors.rs               # Error handling
├── python/
│   └── automation_bridge.py    # Python interface
├── DYNAMIC_MODEL_ROUTER.md     # Model routing docs
└── README.md                   # This file
```

---

## 🎯 Key Features Summary

✅ **Dynamic Model Routing** - Auto-selects best AI model  
✅ **Tri-Cameral Governance** - CIC/AEE/CSF decision making  
✅ **Concurrent Subagents** - Parallel task execution  
✅ **Resource Management** - Billing & quota tracking  
✅ **Cyclic Workflows** - Plan→Analyze→Build→Analyze→Test  
✅ **Race Detection** - Prevents conflicts  
✅ **Change Tracking** - Smart analysis of modifications  
✅ **Multi-Language** - Rust core + Python bridge  
✅ **Modular Design** - Completely separate from SAM-D  

---

## 🚀 Getting Started

### 1. Build Rust Core (Optional)
```bash
cd automation_framework
cargo build --release
```

### 2. Use Python Bridge
```python
from automation_bridge import AutomationFramework

# Framework is ready to use!
framework = AutomationFramework()
```

### 3. Run Example
```bash
cd automation_framework/python
python3 automation_bridge.py
```

---

## 🎓 Next Steps

The automation framework is **complete and ready**! It can now be used to:

1. **Build SAM-D Phase 2** (Power/Control systems)
2. **Orchestrate any development workflow**
3. **Automatically optimize AI model usage**
4. **Ensure governance and safety**

**To use with SAM-D**:
```python
# Use this automation framework to build SAM-D
# They are modular - this is the builder, SAM-D is the product
```

---

**Status**: ✅ **COMPLETE & OPERATIONAL**  
**Modularity**: ✅ **Standalone from SAM-D**  
**Performance**: ✅ **Rust core for speed/security**  
**Flexibility**: ✅ **Python bridge for ease**  
**Intelligence**: ✅ **Dynamic model routing**  
