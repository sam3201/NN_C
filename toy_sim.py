import numpy as np


class EpistemicSim:
    """
    Minimal K/U/Omega growth simulator with regime-aware control.
    K: structured knowledge
    U: explicit unknowns
    O: opacity/cryptic frontier (not yet representable)
    """

    def __init__(self, K=1.0, U=5.0, O=10.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.K = float(K)
        self.U = float(U)
        self.O = float(O)

        # Core coefficients (tune these)
        self.alpha = 0.05  # discovery strength
        self.beta = 1.10  # discovery scaling with K
        self.gamma = 0.02  # maintenance burden
        self.delta = 1.00  # burden scaling

        self.lmbd_contra = 0.01  # contradiction penalty strength

        # Unknown expansion / resolution
        self.eta = 0.03  # new unknowns created by expanding knowledge
        self.mu = 1.0
        self.kappa = 0.04  # resolution rate

        # Opacity expansion / morphogenesis
        self.xi = 0.02  # new opacity created by pushing boundaries
        self.nu = 1.0
        self.chi = 0.06  # morphogenesis conversion rate

        # Control knobs (these are like m_t projections)
        self.research_effort = 0.5  # [0..1]
        self.verify_effort = 0.5  # [0..1]
        self.morph_effort = 0.2  # [0..1]

        # Simple memory of recent progress for plateau detection
        self.K_hist = []

    def sigma_frontier(self):
        # Frontier fuel saturation
        rho = 0.7
        return (self.U + rho * self.O) / (1.0 + self.U + rho * self.O)

    def contradiction(self):
        # crude: contradiction rises if U and O are large relative to K
        return max(0.0, (self.U + self.O) / (1.0 + self.K) - 1.0)

    def step(self, dt=1.0):
        K, U, O = self.K, self.U, self.O

        sigma = self.sigma_frontier()
        contra = self.contradiction()

        # Discovery term (research effort boosts it)
        discovery = self.alpha * (K**self.beta) * sigma * (0.5 + self.research_effort)

        # Maintenance burden (verification effort reduces penalty)
        burden = self.gamma * (K**self.delta) * (1.2 - 0.7 * self.verify_effort)

        # Contradiction penalty
        contra_pen = self.lmbd_contra * (K**self.delta) * contra

        dK = (discovery - burden - contra_pen) * dt

        # Unknowns: created by knowledge expansion, reduced by verification + research resolution
        created_U = (
            self.eta
            * (max(K, 1e-9) ** self.mu)
            * (0.4 + 0.6 * self.research_effort)
            * dt
        )
        resolved_U = self.kappa * U * (0.3 + 0.7 * self.verify_effort) * dt
        dU = created_U - resolved_U

        # Opacity: created when pushing frontier; reduced by morphogenesis effort
        created_O = (
            self.xi
            * (max(K, 1e-9) ** self.nu)
            * (0.5 + 0.5 * self.research_effort)
            * dt
        )
        morphed_O = self.chi * O * (0.2 + 0.8 * self.morph_effort) * dt
        dO = created_O - morphed_O

        # Clamp to keep values sane
        self.K = max(0.0, K + dK)
        self.U = max(0.0, U + dU)
        self.O = max(0.0, O + dO)

        self.K_hist.append(self.K)
        if len(self.K_hist) > 50:
            self.K_hist.pop(0)

        return {
            "K": self.K,
            "U": self.U,
            "O": self.O,
            "dK": dK,
            "dU": dU,
            "dO": dO,
            "sigma": sigma,
            "contra": contra,
        }

    def plateau(self, window=20, eps=1e-3):
        if len(self.K_hist) < window:
            return False
        a = np.mean(self.K_hist[: window // 2])
        b = np.mean(self.K_hist[window // 2 :])
        return (b - a) < eps

    def control_update(self):
        """
        A tiny stand-in for your m_t -> regime controller:
        - If plateau: increase morphogenesis + research
        - If contradiction high: increase verify
        """
        contra = self.contradiction()
        if self.plateau():
            self.morph_effort = min(1.0, self.morph_effort + 0.10)
            self.research_effort = min(1.0, self.research_effort + 0.05)
        if contra > 0.5:
            self.verify_effort = min(1.0, self.verify_effort + 0.05)
        # mild decay back toward baseline
        self.research_effort = 0.98 * self.research_effort + 0.02 * 0.5
        self.verify_effort = 0.98 * self.verify_effort + 0.02 * 0.5
        self.morph_effort = 0.98 * self.morph_effort + 0.02 * 0.2


def run():
    sim = EpistemicSim(K=1.0, U=4.0, O=12.0, seed=1)
    for t in range(200):
        sim.control_update()
        out = sim.step(dt=1.0)
        if t % 20 == 0:
            print(
                t,
                {k: round(out[k], 3) for k in ["K", "U", "O", "dK", "sigma", "contra"]},
            )


if __name__ == "__main__":
    run()
