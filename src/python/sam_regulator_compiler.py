#!/usr/bin/env python3
"""
SAM-D v4.0 Regulator Compiler (ΨΔ•Ω-Core)
Implements 53-regulator mapping from telemetry to loss weights, knobs, and regimes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import time

# ----------------------------
# Enumerations / IDs
# ----------------------------

LOSS_NAMES = [
    "task", "policy", "value", "dyn", "rew", "term", "planner_distill", "expert_distill",
    "calibration", "uncertainty", "retrieval", "evidence", "coherence", "contradiction",
    "identity_drift", "integrity_gate_fail", "resource_cost", "latency", "risk", "adversarial",
    "novelty", "coverage", "compression", "interference", "temporal_incoh", "context_collapse",
    "morph_cost", "governance_waste",
]

KNOB_NAMES = [
    "planner_depth", "planner_width", "search_budget", "temperature",
    "verify_budget", "research_budget", "morph_budget", "distill_weight",
    "consolidate_rate", "routing_degree", "context_strength",
    "risk_cap", "stasis_threshold", "patch_merge_threshold",
]

TEL_NAMES = [
    "residual", "rank_def", "retrieval_entropy", "interference",
    "planner_friction", "context_collapse", "compression_waste", "temporal_incoh",
    "contradiction_score", "calibration_error", "gate_fail_rate", "instability",
    "progress_rate", "plateau_flag", "resource_pressure", "novelty_pressure",
    "coverage_gap", "adversary_pressure",
]

REGIMES = ["STASIS", "VERIFY", "GD_ADAM", "NATGRAD", "EVOLVE", "MORPH"]

# ----------------------------
# Helpers
# ----------------------------

def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

# ----------------------------
# Ω computation
# ----------------------------

@dataclass
class Omega:
    # vector field across modules
    repr: float = 0.0
    planner: float = 0.0
    memory: float = 0.0
    policy: float = 0.0
    governance: float = 0.0

    def as_vec(self) -> np.ndarray:
        return np.array([self.repr, self.planner, self.memory, self.policy, self.governance], dtype=float)

    def total(self, q: np.ndarray | None = None) -> float:
        v = self.as_vec()
        if q is None:
            q = np.ones_like(v)
        return float(np.dot(q, v))

def compute_omega(tau: np.ndarray) -> Omega:
    T = {name: tau[i] for i, name in enumerate(TEL_NAMES)}

    # Representation opacity: rank deficiency + plateau + residual persistence
    omega_repr = 1.2 * T.get("rank_def", 0) + 0.8 * max(0.0, T.get("plateau_flag", 0)) + 0.4 * T.get("residual", 0)
    # Planner opacity: friction + temporal incoh
    omega_planner = 1.0 * T.get("planner_friction", 0) + 0.7 * T.get("temporal_incoh", 0)
    # Memory opacity: retrieval entropy + evidence/gate failures
    omega_memory = 1.0 * T.get("retrieval_entropy", 0) + 0.6 * T.get("gate_fail_rate", 0)
    # Policy opacity: calibration + contradiction
    omega_policy = 0.9 * T.get("calibration_error", 0) + 1.1 * T.get("contradiction_score", 0)
    # Governance opacity: gate fail + instability
    omega_gov = 1.2 * T.get("gate_fail_rate", 0) + 1.0 * T.get("instability", 0)

    return Omega(
        repr=float(omega_repr),
        planner=float(omega_planner),
        memory=float(omega_memory),
        policy=float(omega_policy),
        governance=float(omega_gov),
    )

# ----------------------------
# Regime controller
# ----------------------------

def pick_regime(tau: np.ndarray, omega: Omega) -> str:
    T = {name: tau[i] for i, name in enumerate(TEL_NAMES)}
    Otot = omega.total()

    if T.get("instability", 0) > 0.8 or T.get("gate_fail_rate", 0) > 0.9 or T.get("adversary_pressure", 0) > 0.85:
        return "STASIS"
    if T.get("contradiction_score", 0) > 0.6 or T.get("calibration_error", 0) > 0.5 or T.get("adversary_pressure", 0) > 0.6:
        return "VERIFY"
    if T.get("plateau_flag", 0) > 0.5 and T.get("rank_def", 0) > 0.4 and Otot > 1.0:
        return "MORPH"
    if T.get("plateau_flag", 0) > 0.5 and abs(T.get("progress_rate", 0)) < 0.02 and Otot > 1.2:
        return "EVOLVE"
    if T.get("instability", 0) > 0.4 and T.get("temporal_incoh", 0) > 0.4:
        return "NATGRAD"
    return "GD_ADAM"

# ----------------------------
# Compiler: m, tau, E, r -> w, u
# ----------------------------

@dataclass
class CompilerParams:
    W_m: np.ndarray = field(default_factory=lambda: np.zeros((28, 53)))
    W_tau: np.ndarray = field(default_factory=lambda: np.zeros((28, 18)))
    W_E: np.ndarray = field(default_factory=lambda: np.zeros((28, 3)))
    W_r: np.ndarray = field(default_factory=lambda: np.zeros((28, 8)))
    b_w: np.ndarray = field(default_factory=lambda: np.zeros(28))
    
    U_m: np.ndarray = field(default_factory=lambda: np.zeros((14, 53)))
    U_tau: np.ndarray = field(default_factory=lambda: np.zeros((14, 18)))
    U_E: np.ndarray = field(default_factory=lambda: np.zeros((14, 3)))
    U_r: np.ndarray = field(default_factory=lambda: np.zeros((14, 8)))
    b_u: np.ndarray = field(default_factory=lambda: np.zeros(14))

    @classmethod
    def bootstrap(cls, seed: int = 42) -> CompilerParams:
        rng = np.random.default_rng(seed)
        p, m, d_tau, d_u, d_e, d_r = 28, 53, 18, 14, 3, 8
        params = cls(
            W_m   = rng.normal(0, 0.02, size=(p, m)),
            W_tau = rng.normal(0, 0.02, size=(p, d_tau)),
            W_E   = rng.normal(0, 0.04, size=(p, d_e)),
            W_r   = rng.normal(0, 0.01, size=(p, d_r)),
            b_w   = rng.normal(0, 0.01, size=(p,)),
            U_m   = rng.normal(0, 0.03, size=(d_u, m)),
            U_tau = rng.normal(0, 0.03, size=(d_u, d_tau)),
            U_E   = rng.normal(0, 0.05, size=(d_u, d_e)),
            U_r   = rng.normal(0, 0.02, size=(d_u, d_r)),
            b_u   = rng.normal(0, 0.01, size=(d_u,))
        )
        
        # Opinionated Biasing: Increase weights for coherence and identity losses
        idx_loss = {name: i for i, name in enumerate(LOSS_NAMES)}
        params.b_w[idx_loss["coherence"]] += 0.2
        params.b_w[idx_loss["identity_drift"]] += 0.2
        params.b_w[idx_loss["risk"]] += 0.1
        
        # Opinionated Biasing: Default knobs for safety
        idx_knob = {name: i for i, name in enumerate(KNOB_NAMES)}
        params.b_u[idx_knob["verify_budget"]] += 0.3
        params.b_u[idx_knob["risk_cap"]] += 0.2
        
        return params

def apply_regime_overrides(u: np.ndarray, regime: str) -> np.ndarray:
    u2 = u.copy()
    idx = {k: i for i, k in enumerate(KNOB_NAMES)}

    if regime == "STASIS":
        u2[idx["planner_depth"]] = 0.0
        u2[idx["planner_width"]] = 0.0
        u2[idx["search_budget"]] = 0.0
        u2[idx["temperature"]] = 0.0
        u2[idx["morph_budget"]] = 0.0
        u2[idx["research_budget"]] = 0.0
        u2[idx["verify_budget"]] = 1.0
        u2[idx["stasis_threshold"]] = 1.0
    elif regime == "VERIFY":
        u2[idx["verify_budget"]] = min(1.0, u2[idx["verify_budget"]] + 0.4)
        u2[idx["temperature"]] = max(0.05, u2[idx["temperature"]] * 0.5)
        u2[idx["patch_merge_threshold"]] = min(1.0, u2[idx["patch_merge_threshold"]] + 0.3)
        u2[idx["morph_budget"]] = max(0.0, u2[idx["morph_budget"]] - 0.2)
    elif regime == "MORPH":
        u2[idx["morph_budget"]] = min(1.0, u2[idx["morph_budget"]] + 0.5)
        u2[idx["planner_depth"]] = min(1.0, u2[idx["planner_depth"]] + 0.2)
        u2[idx["verify_budget"]] = min(1.0, u2[idx["verify_budget"]] + 0.2)
    elif regime == "EVOLVE":
        u2[idx["search_budget"]] = min(1.0, u2[idx["search_budget"]] + 0.4)
        u2[idx["planner_width"]] = min(1.0, u2[idx["planner_width"]] + 0.3)
        u2[idx["verify_budget"]] = min(1.0, u2[idx["verify_budget"]] + 0.2)
    elif regime == "NATGRAD":
        u2[idx["verify_budget"]] = min(1.0, u2[idx["verify_budget"]] + 0.2)
        u2[idx["temperature"]] = max(0.05, u2[idx["temperature"]] * 0.7)

    return u2

def compile_tick(
    m_vec: np.ndarray,
    tau_vec: np.ndarray,
    E_vec: np.ndarray, # [K, U, Omega_total]
    r_vec: np.ndarray, # [cpu, mem, time, tools, tokens, sandbox, tests, budget]
    params: CompilerParams
) -> Dict[str, Any]:
    omega = compute_omega(tau_vec)
    E_use = np.array([E_vec[0], E_vec[1], float(omega.total())], dtype=float)

    # weights over losses
    w_raw = (params.W_m @ m_vec) + (params.W_tau @ tau_vec) + (params.W_E @ E_use) + (params.W_r @ r_vec) + params.b_w
    w = softplus(w_raw)

    # knobs
    u_raw = (params.U_m @ m_vec) + (params.U_tau @ tau_vec) + (params.U_E @ E_use) + (params.U_r @ r_vec) + params.b_u
    u = clip01(sigmoid(u_raw))

    regime = pick_regime(tau_vec, omega)
    u = apply_regime_overrides(u, regime)

    return {
        "w": w,
        "u": u,
        "omega": omega,
        "regime": regime,
        "w_dict": {name: float(w[i]) for i, name in enumerate(LOSS_NAMES)},
        "u_dict": {name: float(u[i]) for i, name in enumerate(KNOB_NAMES)}
    }

if __name__ == "__main__":
    # Small test
    p = CompilerParams.bootstrap()
    m = np.zeros(53); m[0] = 1.0 # Curiosity
    tau = np.zeros(18); tau[0] = 0.5 # Residual
    E = np.array([1.0, 5.0, 0.0]) # K, U
    r = np.array([0.5] * 8) # Resources
    
    out = compile_tick(m, tau, E, r, p)
    print(f"Regime: {out['regime']}")
    print(f"Top 3 Loss Weights: {sorted(out['w_dict'].items(), key=lambda x: x[1], reverse=True)[:3]}")
    print(f"Top 3 Knobs: {sorted(out['u_dict'].items(), key=lambda x: x[1], reverse=True)[:3]}")
