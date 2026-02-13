# SAM-D AGI (ΨΔ•Ω-Core v5.0.0 Recursive)

SAM-D is a hybrid Python/C recursive meta-evolutionary system with a web dashboard, slash-command interface, and C-accelerated cores for meta-control and dual-system simulation. It includes ΨΔ•Ω-Core morphogenesis and a PDI-T submodel lifecycle.

## What’s Included
- Main orchestrator + API server + UI: `src/python/complete_sam_unified.py`
- Web dashboard + terminal UI served by the API
- C extensions for performance (built via `setup.py`)
- Deterministic bootstrapping via Make / Docker / CI

---

## System Requirements

### Supported Python Versions
- **Python 3.10+** (recommended: **3.12 or 3.13**)
- Avoid Python **3.14+** if your platform’s scientific wheels aren’t available yet (NumPy/Pillow can fall back to source builds).

Check your version:
```bash
python --version

