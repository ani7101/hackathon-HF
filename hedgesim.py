#!/usr/bin/env python3
"""
hedgesim.py -- the simulator behind level 5.

You have SOLD one European call (strike K, maturity T) and received the Black-Scholes premium.
You hedge by holding delta shares of the stock. Each time you trade you pay a proportional cost
c * price * |shares traded|. At maturity the hedge is closed to what settlement needs (1 share if
the option finished in the money, else 0), and that last trade pays the cost too.

Policy (what the model chooses):
    interval  m     look at the hedge only every m-th time step
    band      b     at a look, trade only if |held delta - target delta| > b (then trade to target)
  or, on 5.3, b can depend on the state: band(gamma, S, tau).

Score of a policy on a set of paths:
    J = E[total cost] + lam * std(terminal P&L)          (lower is better)

Paths use antithetic normals and a fixed seed, so J is deterministic: the same policy always
gets the same number, and different path sets (practice vs validation) are separate seeds.
"""

import math

import numpy as np

from hedge_core import N, bs_call


def default_params(**kw):
    prm = dict(S0=100.0, K=100.0, T=1.0, r=0.0, sigma=0.2, c=0.005, lam=1.0, steps=252)
    prm.update(kw)
    prm["steps"] = int(prm["steps"])
    return prm


class PathSet:
    """Price paths plus everything that depends only on the path (delta, gamma), computed once,
    so that evaluating many policies on the same paths is cheap."""

    def __init__(self, prm, paths, seed):
        self.prm = dict(prm)
        P = paths - paths % 2
        n, T = prm["steps"], prm["T"]
        dt = T / n
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((P // 2, n))
        z = np.concatenate([z, -z], axis=0)                      # antithetic
        sig = prm["sigma"]
        inc = (prm["r"] - 0.5 * sig ** 2) * dt + sig * math.sqrt(dt) * z
        logS = np.concatenate([np.zeros((P, 1)), np.cumsum(inc, axis=1)], axis=1)
        self.S = prm["S0"] * np.exp(logS)                        # (P, n+1)
        self.P, self.n, self.dt = P, n, dt
        self.tau = T - dt * np.arange(n)                         # time left at each decision
        S = self.S[:, :n]
        sT = sig * np.sqrt(self.tau)
        d1 = (np.log(S / prm["K"]) + (prm["r"] + 0.5 * sig ** 2) * self.tau) / sT
        self.delta = N(d1)
        self.gamma = np.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi) / (S * sT)
        self.premium = float(bs_call(prm["S0"], prm["K"], T, prm["r"], sig))
        self._band_cache = {}

    # ------------------------------------------------------------------ the hedge

    def run(self, interval=1, band=0.0):
        """Simulate the hedge. `band` is a number, or an array of shape (P, n)."""
        prm, P, n = self.prm, self.P, self.n
        c, K = prm["c"], prm["K"]
        g = math.exp(prm["r"] * self.dt)
        scalar = np.isscalar(band)
        h = np.zeros(P)
        cash = np.full(P, self.premium)
        cost = np.zeros(P)
        trades = np.zeros(P)
        for i in range(n):
            if i > 0:
                cash *= g
            if i == 0:
                trade = self.delta[:, 0].copy()                  # set up the initial hedge
            elif i % interval == 0:
                diff = self.delta[:, i] - h
                b = band if scalar else band[:, i]
                trade = np.where(np.abs(diff) > b, diff, 0.0)
            else:
                continue
            S = self.S[:, i]
            fee = c * S * np.abs(trade)
            cash -= trade * S + fee
            cost += fee
            trades += (trade != 0)
            h += trade
        cash *= g
        ST = self.S[:, n]
        target = (ST > K).astype(float)                          # what settlement needs
        trade = target - h
        fee = c * ST * np.abs(trade)
        cash -= trade * ST + fee
        cost += fee
        trades += (trade != 0)
        pnl = cash + target * ST - np.maximum(ST - K, 0.0)
        return self._summary(pnl, cost, trades)

    def unhedged(self):
        ST = self.S[:, self.n]
        pnl = self.premium * math.exp(self.prm["r"] * self.prm["T"]) - np.maximum(ST - self.prm["K"], 0.0)
        return self._summary(pnl, np.zeros(self.P), np.zeros(self.P))

    def _summary(self, pnl, cost, trades):
        risk = float(pnl.std())
        mc = float(cost.mean())
        return dict(J=mc + self.prm["lam"] * risk, cost=mc, risk=risk,
                    trades=float(trades.mean()), pnl_mean=float(pnl.mean()))

    # ------------------------------------------------------------- state-dependent band

    def band_from(self, fn):
        """Evaluate a compiled band(gamma, S, tau) formula on every (path, step). Returns an array
        of shape (P, n) clipped to [0, 1], or None if the formula produced anything non-finite."""
        prm = self.prm
        env = dict(gamma=self.gamma, S=self.S[:, :self.n], tau=np.broadcast_to(self.tau, self.gamma.shape),
                   c=prm["c"], lam=prm["lam"], sigma=prm["sigma"], r=prm["r"], T=prm["T"])
        try:
            b = np.asarray(fn(**env), dtype=float)
        except (ArithmeticError, ValueError, TypeError):
            return None
        try:
            b = np.broadcast_to(b, self.gamma.shape)
        except ValueError:
            return None
        if not np.isfinite(b).all():
            return None
        return np.clip(b, 0.0, 1.0)


# ------------------------------------------------------------------------ reference solver

INTERVALS = tuple(range(1, 64))
BANDS = tuple(round(0.005 * k, 3) for k in range(0, 101))       # 0 .. 0.50
KAPPAS = tuple(round(0.1 * k, 2) for k in range(1, 81))          # 0.1 .. 8


def kappa_band(ps, kappa):
    """The structured family  band = kappa * (c * S * gamma^2)^(1/3)  (Whalley-Wilmott scaling)."""
    if "kappa_base" not in ps._band_cache:
        ps._band_cache["kappa_base"] = (ps.prm["c"] * ps.S[:, :ps.n] * ps.gamma ** 2) ** (1.0 / 3.0)
    return kappa * ps._band_cache["kappa_base"]


def best_policy(ps, family):
    """Grid search on a path set. family: 'interval' | 'band' | 'formula'. Returns (J*, policy, summary)."""
    best = None
    if family == "interval":
        cands = [(dict(interval=m, band=0.0), ps.run(m, 0.0)) for m in INTERVALS if m < ps.n]
    elif family == "band":
        cands = [(dict(interval=1, band=b), ps.run(1, b)) for b in BANDS]
    elif family == "formula":
        cands = [(dict(interval=1, band=b), ps.run(1, b)) for b in BANDS]
        cands += [(dict(kappa=k), ps.run(1, np.clip(kappa_band(ps, k), 0, 1))) for k in KAPPAS]
    else:
        raise ValueError(family)
    for pol, s in cands:
        if best is None or s["J"] < best[2]["J"]:
            best = (s["J"], pol, s)
    return best


if __name__ == "__main__":
    # exploration: how does the cost/risk trade-off look?
    import itertools
    import time
    for c, lam in itertools.product((0.002, 0.005, 0.01), (1.0, 2.0)):
        prm = default_params(c=c, lam=lam)
        t0 = time.perf_counter()
        ps = PathSet(prm, 20000, seed=11)
        naive, unh = ps.run(1, 0.0), ps.unhedged()
        bi = best_policy(ps, "interval")
        bb = best_policy(ps, "band")
        print(f"c={c} lam={lam}: unhedged J={unh['J']:.2f}  naive J={naive['J']:.3f} "
              f"(cost {naive['cost']:.3f}, risk {naive['risk']:.3f}, trades {naive['trades']:.0f})")
        print(f"    best interval {bi[1]['interval']:>2}: J={bi[0]:.3f} ({(naive['J'] / bi[0] - 1) * 100:.0f}% better than naive)"
              f"   best band {bb[1]['band']:.3f}: J={bb[0]:.3f} ({(naive['J'] / bb[0] - 1) * 100:.0f}% better)"
              f"   [{time.perf_counter() - t0:.1f}s]")
