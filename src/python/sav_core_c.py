"""SAV core adapter backed by sam_sav_dual_system.

This provides a stable API for SAM_AGI while using the dual-system arena
internally for adversarial pressure testing.
"""

from __future__ import annotations

import sam_sav_dual_system as _dual

DEFAULT_STATE_DIM = 16
DEFAULT_ARENA_DIM = 4


def create(*args):
    """Create an SAV core arena.

    Supported call patterns:
    - create() -> defaults
    - create(seed)
    - create(state_dim, arena_dim)
    - create(state_dim, arena_dim, seed)
    """
    if len(args) == 0:
        return _dual.create(DEFAULT_STATE_DIM, DEFAULT_ARENA_DIM, 0)
    if len(args) == 1:
        seed = int(args[0])
        return _dual.create(DEFAULT_STATE_DIM, DEFAULT_ARENA_DIM, seed)
    if len(args) == 2:
        state_dim, arena_dim = args
        return _dual.create(int(state_dim), int(arena_dim), 0)
    state_dim, arena_dim, seed = args[:3]
    return _dual.create(int(state_dim), int(arena_dim), int(seed))


def step(arena, *args):
    """Advance the arena one step. Extra args are ignored for compatibility."""
    return _dual.step(arena)


def run(arena, steps):
    """Run the arena for N steps."""
    return _dual.run(arena, int(steps))


def get_state(arena):
    """Return full arena state."""
    return _dual.get_state(arena)


def get_status(arena):
    """Return a concise status payload."""
    state = _dual.get_state(arena)
    return {
        "sam_alive": bool(state.get("sam_alive", False)),
        "sav_alive": bool(state.get("sav_alive", False)),
        "sam_survival": state.get("sam_survival", 0.0),
        "sav_survival": state.get("sav_survival", 0.0),
        "sam_score": state.get("sam_score", 0.0),
        "sav_score": state.get("sav_score", 0.0),
    }


def force_mutation(arena, target, rounds=1):
    return _dual.force_mutation(arena, int(target), int(rounds))
