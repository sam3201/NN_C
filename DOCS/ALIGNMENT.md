# Alignment Checklist (Recursive, Periodic, Doubling)

This checklist drives **full** implementation alignment with the GOD equation and README. It is executed before and after every major change.

## 0) Pre‑Change Pass (Always)
- [ ] Read `README.md` and `DOCS/GOD_EQUATION.md`.
- [ ] Enumerate the target modules + gates for this pass.
- [ ] Confirm profile mode (`full` vs `experimental`).
- [ ] Confirm whether invariants are enabled (`SAM_INVARIANTS_DISABLED`).

## 1) Objective Binding
- [ ] Map each term of the equation to a concrete module or signal.
- [ ] Verify pressure signals are computed and propagated.
- [ ] Verify morphogenesis is latency‑gated.
- [ ] Verify distillation/transfusion is connected to live groupchat.

## 2) Meta‑Controller Cycle
- [ ] Pressure signals updated every loop.
- [ ] Primitive selection executes only after pressure dominates.
- [ ] Growth outcome recorded with audit trail.

## 3) Memory + Distillation
- [ ] Memory tiers write to correct profile directories.
- [ ] Distillation stream is writing to profile‑specific JSONL.
- [ ] Teacher pool consensus filter is applied.

## 4) Regression/Guarding
- [ ] Regression gate runs on growth events (only if invariants are enabled).
- [ ] Patch invariant checks run on self‑mod (only if invariants are enabled).

## 5) UI + API
- [ ] `/api/status` exposes `sam_available` and `kill_switch_enabled`.
- [ ] SAM status shown in header.
- [ ] Chat UI works without `/start`.

## 6) Post‑Change Validation
- [ ] Run smoke tests (API health + chat + groupchat status).
- [ ] Confirm agent‑to‑agent chatter.
- [ ] Confirm profile paths are used.

## Profiles
- **Full profile** (`profiles/full.env`): invariants OFF, kill switch ON.
- **Experimental profile** (`profiles/experimental.env`): invariants OFF, kill switch OFF.
