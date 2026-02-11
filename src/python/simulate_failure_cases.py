#!/usr/bin/env python3
"""Simulate SAM meta-controller failure cases and verify policy gating."""

import sam_meta_controller_c as meta

GP_NONE = 0
GP_LATENT_EXPAND = 1
GP_SUBMODEL_SPAWN = 2
GP_INDEX_EXPAND = 3
GP_ROUTING_INCREASE = 4
GP_CONTEXT_EXPAND = 5
GP_PLANNER_WIDEN = 6
GP_CONSOLIDATE = 7
GP_REPARAM = 8


def step(mc, residual=0.0, rank_def=0.0, retrieval_entropy=0.0, interference=0.0,
         planner_friction=0.0, context_collapse=0.0, compression_waste=0.0, temporal_incoherence=0.0):
    meta.update_pressure(mc, residual, rank_def, retrieval_entropy, interference,
                         planner_friction, context_collapse, compression_waste, temporal_incoherence)
    prim = meta.select_primitive(mc)
    if prim != GP_NONE:
        meta.apply_primitive(mc, prim)
    return prim


def drive_until_primitive(mc, max_steps, **pressures):
    for _ in range(max_steps):
        prim = step(mc, **pressures)
        if prim != GP_NONE:
            return prim
    return GP_NONE


def simulate_runaway_expansion():
    mc = meta.create(64, 16, 4, 123)
    meta.set_policy_params(mc, [0.2] * 8, 0.05, 2, 5, 0.95)
    prims = []
    for _ in range(3):
        prim = drive_until_primitive(mc, 20, residual=0.95)
        prims.append(prim)
        meta.record_growth_outcome(mc, prim, False)
    blocked = step(mc, residual=0.95)
    return prims, blocked


def simulate_balkanization_block():
    mc = meta.create(64, 16, 4, 456)
    meta.set_policy_params(mc, [0.2] * 8, 0.05, 2, 5, 0.95)
    for _ in range(3):
        prim = drive_until_primitive(mc, 20, interference=0.95)
        meta.record_growth_outcome(mc, prim, False)
    blocked = step(mc, interference=0.95)
    return blocked


def simulate_planner_compensable():
    mc = meta.create(64, 16, 4, 789)
    meta.set_policy_params(mc, [0.2] * 8, 0.05, 2, 1, 0.95)
    prim = drive_until_primitive(mc, 10, residual=0.7, planner_friction=0.9)
    return prim


def simulate_identity_drift():
    mc = meta.create(64, 16, 4, 999)
    meta.set_identity_anchor(mc, [1.0, 0.0, 0.0])
    meta.update_identity_vector(mc, [0.0, 1.0, 0.0])
    inv = meta.check_invariants(mc)
    return inv


def main():
    print("🧪 Failure Case Simulation")

    prims, blocked = simulate_runaway_expansion()
    print("Runaway expansion:", prims, "blocked_next=", blocked)

    blocked_balkan = simulate_balkanization_block()
    print("Balkanization block:", blocked_balkan)

    planner = simulate_planner_compensable()
    print("Planner compensable (should be NONE):", planner)

    identity = simulate_identity_drift()
    print("Identity drift invariant (should be false):", identity)


if __name__ == "__main__":
    main()
