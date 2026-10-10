#!/usr/bin/env python3
"""
level5_hedge.py -- Level 5: hedge an option when trading costs money.

Black-Scholes says a short option can be hedged perfectly by trading continuously for free. Here
every trade costs a fraction c of its value. Trade rarely and the hedge drifts (risk). Trade
often and costs pile up. The model picks the policy; the checker simulates it on fixed validation paths.

    J = E[total cost] + lam * std(terminal P&L)        lower is better

Sub-problems:
    5.1  choose an interval m   (look at the hedge every m-th step)           one number
    5.2  choose a band b        (trade only if delta has drifted by more than b)   one number
    5.3  choose a band FORMULA  band(gamma, S, tau)  -- can structure beat a constant?

Reward, out of 1.0, each 0.2:  valid answer / beats not hedging / beats the obvious daily full
hedge / within 10% of the best reference-grid J / within 3% of it (5.3 uses 5% and 1%).

The feedback is directional: it says whether you trade too much or too little, never the
optimum. The model can try ideas on 2000 practice paths with  SIM:  but is graded on 20000
different ones, so tuning to noise does not pay.

    python level5_hedge.py --list
    python level5_hedge.py --show 2 --answer
    python level5_hedge.py --selftest
"""

import argparse
import random
import re
import sys

import numpy as np

import hedgesim
from hedgesim import PathSet, best_policy, default_params, kappa_band
from hedge_core import BadExpression, compile_expr, last_line

LEVEL = 5
TITLE = "hedging a short call under transaction costs"
SUBS = (1, 2, 3)
FAMILY = {1: "interval", 2: "band", 3: "formula"}
TEST_PATHS, TRAIN_PATHS = 20_000, 2_000
TEST_SEED, TRAIN_SEED = 9001, 101
WEIGHTS = dict(valid=0.2, beats_unhedged=0.2, beats_naive=0.2, within_loose=0.2, within_tight=0.2)
BAND_VARS = ["gamma", "S", "tau", "c", "lam", "sigma", "r", "T"]

_CACHE = {}


# ------------------------------------------------------------------------------ problems


def make(sub, seed=0):
    if sub not in SUBS:
        raise ValueError(f"level 5 has sub-problems {SUBS}")
    r = random.Random(7000 + seed * 100 + sub)
    # c is kept >= 0.005 on purpose: below that, daily hedging is nearly optimal and the
    # level has nothing to teach (measured: 5-10% gap at c = 0.002, 40-100% at c >= 0.005).
    T = r.choice([0.5, 1.0])
    prm = default_params(c=r.choice([0.005, 0.0075, 0.01]), lam=r.choice([1.0, 2.0]),
                         sigma=r.choice([0.2, 0.3]), r=r.choice([0.0, 0.02]), T=T,
                         steps=round(252 * T))
    return dict(name=f"level5.{sub}", level=5, sub=sub, seed=seed, prm=prm,
                family=FAMILY[sub], hint=1)


def _key(prm):
    return tuple(sorted(prm.items()))


def test_set(p):
    k = ("test", _key(p["prm"]))
    if k not in _CACHE:
        _CACHE[k] = PathSet(p["prm"], TEST_PATHS, TEST_SEED)
    return _CACHE[k]


def train_set(p):
    k = ("train", _key(p["prm"]))
    if k not in _CACHE:
        _CACHE[k] = PathSet(p["prm"], TRAIN_PATHS, TRAIN_SEED)
    return _CACHE[k]


def reference(p):
    """Best reference-grid J on the fixed validation paths, plus the two baselines. Cached."""
    k = ("ref", _key(p["prm"]), p["family"])
    if k not in _CACHE:
        ps = test_set(p)
        Jstar, pol, summ = best_policy(ps, p["family"])
        _CACHE[k] = dict(Jstar=Jstar, policy=pol, summary=summ, naive=ps.run(1, 0.0),
                         unhedged=ps.unhedged())
    return _CACHE[k]


# -------------------------------------------------------------------------------- prompt


def prompt_of(p):
    prm, sub = p["prm"], p["sub"]
    steps = prm["steps"]
    head = [
        "You have SOLD one European call option and must hedge it.",
        "",
        f"The stock starts at {prm['S0']:g}; the strike is {prm['K']:g}; maturity is T = {prm['T']:g} "
        f"years; the risk-free rate is r = {prm['r']:g}; the volatility is sigma = {prm['sigma']:g} "
        f"and it is the same volatility the option was priced with, so a free, continuous hedge "
        f"would leave exactly zero risk and zero profit. You received the Black-Scholes premium up front.",
        "",
        f"You hedge by holding shares of the stock. Time is cut into {steps} equal decision times. "
        f"At a decision time you may trade to the Black-Scholes delta. Every trade costs "
        f"c = {prm['c']:g} of the traded value (c x price x number of shares). At maturity "
        f"your shares are settled to what delivery needs, and that last trade pays the cost too.",
        "",
        f"Your policy is scored on many simulated price paths you cannot see:",
        f"    J = E[total transaction cost] + lam x std(terminal profit and loss),   lam = {prm['lam']:g}",
        "Lower is better. Trading more often lowers the std but raises the cost; the aim is the "
        "best balance.",
        "",
    ]
    if sub == 1:
        body = [
            "YOUR CHOICE: an integer interval m between 1 and 63. You look at your hedge only at "
            "every m-th decision time (and at the start); when you look, you trade to the "
            "Black-Scholes delta.",
            "",
            "Finish your reply with exactly one line:",
            "  policy: interval=<integer>",
            "Worked example of the format:",
            "  policy: interval=5",
        ]
    elif sub == 2:
        body = [
            "YOUR CHOICE: a no-trade band b between 0 and 0.5. You look at your hedge at every "
            "decision time, and trade to the Black-Scholes delta only if the number of shares you "
            "hold differs from it by MORE than b; otherwise you do nothing.",
            "",
            "Finish your reply with exactly one line:",
            "  policy: band=<number>",
            "Worked example of the format:",
            "  policy: band=0.03",
        ]
    else:
        body = [
            "YOUR CHOICE: a no-trade band that may depend on the state. At every decision time you "
            "trade to the Black-Scholes delta only if your holding differs from it by MORE than "
            "band(gamma, S, tau), where gamma is the option's gamma, S the current stock price and "
            "tau the time left to maturity. The result is clipped to [0, 1].",
            "",
            "Allowed: + - * / **, exp, log, sqrt, abs, maximum, minimum, pi, the variables "
            "gamma, S, tau, and the constants c, lam, sigma, r, T (use their names).",
            "",
            "Finish your reply with exactly one line:",
            "  band(gamma,S,tau) = <expression>",
            "Worked example of the format (a constant band):",
            "  band(gamma,S,tau) = 0.03",
        ]
    hint = []
    if p.get("hint", 1) >= 1 and sub == 3:
        hint = ["", "Hint: a band that is the same everywhere is easy; a band that reacts to gamma "
                    "and to the cost c can do better."]
    if p.get("hint", 1) >= 2 and sub == 3:
        hint += ["Second hint: where gamma is large, the delta moves quickly and trading is "
                 "expensive per unit of risk removed. The best band grows with the cost c, "
                 "roughly like a cube root."]
    return "\n".join(head + body + hint)


# ------------------------------------------------------------------------------- parsing


class BadPolicy(Exception):
    pass


def parse_policy(p, text):
    """Returns (interval, band_or_fn, label). band is a float, or a compiled formula for 5.3."""
    sub = p["sub"]
    if sub == 3:
        expr = last_line(text, r"band\s*(?:\([^)]*\))?\s*=\s*(.+)")
        if expr is None:
            raise BadPolicy("No answer line found. End your reply with one line of the form "
                            "band(gamma,S,tau) = <expression>")
        try:
            return 1, compile_expr(expr, BAND_VARS), expr
        except BadExpression as e:
            raise BadPolicy(f"Your band formula could not be read: {e}.")
    line = last_line(text, r"policy\s*:\s*(.+)")
    if line is None:
        raise BadPolicy("No answer line found. End your reply with one line of the form "
                        + ("policy: interval=<integer>" if sub == 1 else "policy: band=<number>"))
    kv = dict(re.findall(r"(interval|band)\s*=\s*([-+0-9.eE]+)", line))
    key = "interval" if sub == 1 else "band"
    if key not in kv:
        raise BadPolicy(f"The answer line must set {key}=<number>.")
    try:
        v = float(kv[key])
    except ValueError:
        raise BadPolicy(f"Could not read {key} as a number.")
    if not np.isfinite(v):
        raise BadPolicy(f"{key} must be finite.")
    if sub == 1:
        if v != int(v) or not 1 <= int(v) <= 63:
            raise BadPolicy("interval must be a whole number between 1 and 63.")
        return int(v), 0.0, f"interval={int(v)}"
    if not 0.0 <= v <= 0.5:
        raise BadPolicy("band must be between 0 and 0.5.")
    return 1, v, f"band={v:g}"


def _run(ps, p, interval, band):
    """Returns the summary, or None when a formula produced non-finite values."""
    if callable(band):
        bm = ps.band_from(band)
        if bm is None:
            return None
        return ps.run(1, bm)
    return ps.run(interval, band)


# ------------------------------------------------------------------------------- checker


def _result(reward, parts, feedback, expr):
    return dict(reward=round(reward, 4), parts=parts, feedback=feedback, expr=expr or "",
                start_error=None)


def check(p, text):
    try:
        interval, band, label = parse_policy(p, text)
    except BadPolicy as e:
        return _result(0.0, {}, str(e), None)
    ps, ref = test_set(p), reference(p)
    s = _run(ps, p, interval, band)
    if s is None:
        return _result(0.0, {}, "Your band formula gave nan or infinity at some (path, time) "
                                "points. Check divisions, logs and square roots.", label)
    J, Js = s["J"], ref["Jstar"]
    loose, tight = (1.05, 1.01) if p["sub"] == 3 else (1.10, 1.03)
    parts = dict(valid=True,
                 beats_unhedged=bool(J < ref["unhedged"]["J"]),
                 beats_naive=bool(J < ref["naive"]["J"] - 1e-12),
                 within_loose=bool(J <= loose * Js),
                 within_tight=bool(J <= tight * Js))
    reward = sum(WEIGHTS[k] for k, v in parts.items() if v)
    if reward >= 0.9999:
        return _result(1.0, parts, "All checks pass.", label)

    best = ref["summary"]
    notes = [f"On {ps.P} fixed validation paths your policy scores J = {J:.3f} "
             f"(expected cost {s['cost']:.3f} + {ps.prm['lam']:g} x risk {s['risk']:.3f}) and "
             f"trades {s['trades']:.0f} times per path."]
    if not parts["beats_unhedged"]:
        notes.append(f"That is worse than not hedging at all (J = {ref['unhedged']['J']:.3f}).")
    if not parts["beats_naive"]:
        notes.append(f"Rebalancing every step to the exact delta would score J = "
                     f"{ref['naive']['J']:.3f}, so your policy is not better than the obvious one.")
    over = s["cost"] > best["cost"] * 1.02
    under = s["risk"] > best["risk"] * 1.02
    if over and not under:
        notes.append("Cost is the problem: you pay more in trading costs than a better policy needs "
                     "to and get no extra risk reduction. Trade less.")
    elif under and not over:
        notes.append("Risk is the problem: your hedge error is larger than a better policy "
                     "achieves while you are not saving enough in costs. Hedge more.")
    elif over and under:
        notes.append("Both cost and risk are worse than a better policy achieves, so this is the "
                     "wrong kind of trade-off, not just too much or too little trading.")
    notes.append(f"J is about {(J / Js - 1) * 100:.0f}% above the best reference grid point on this problem.")
    return _result(reward, parts, " ".join(notes), label)


# --------------------------------------------------------------------------- the SIM tool

SIM_RE = re.compile(r"^\s*SIM:\s*(.+?)\s*$", re.M)

INSTRUCTIONS = (
    "You have a simulator. It runs your policy on 2000 PRACTICE paths and reports J, cost, risk and "
    "trades per path. You are graded on 20000 different paths, so do not over-tune. Write a line "
    "of the form\n"
    "  SIM: {example}\n"
    "You may write up to 4 SIM lines. Stop there and send them; I will reply with the results, "
    "and then you give your final answer."
)


def instructions(p):
    ex = {1: "interval=5", 2: "band=0.03", 3: "band(gamma,S,tau) = 0.03"}[p["sub"]]
    return INSTRUCTIONS.format(example=ex)


FINAL = "Use those results and give your final answer now, as the one line in the required format."


def requests_in(text):
    return [m.group(1).strip() for m in SIM_RE.finditer(text or "")][:4]


def sim(p, ask):
    """Run one SIM line on the practice paths."""
    fake = ask if p["sub"] == 3 else f"policy: {ask}"
    try:
        interval, band, label = parse_policy(p, fake)
    except BadPolicy as e:
        return f"rejected: {e}"
    s = _run(train_set(p), p, interval, band)
    if s is None:
        return "the formula gave nan or infinity somewhere"
    return (f"J = {s['J']:.3f}  (cost {s['cost']:.3f}, risk {s['risk']:.3f}, "
            f"{s['trades']:.0f} trades/path)  [2000 practice paths]")


def answer_block(p, asks):
    return "\n".join(f"SIM: {a}\n  = {sim(p, a)}" for a in asks)


# -------------------------------------------------------------- offline stand-in answers


def _answer_for(p, pol):
    if p["sub"] == 1:
        return f"policy: interval={pol['interval']}"
    if p["sub"] == 2:
        return f"policy: band={pol['band']:g}"
    if "kappa" in pol:
        return f"band(gamma,S,tau) = {pol['kappa']:g}*(c*S*gamma**2)**(1/3)"
    return f"band(gamma,S,tau) = {pol['band']:g}"


def offline_candidates(p):
    ref = reference(p)
    good = [_answer_for(p, ref["policy"])]
    if p["sub"] == 1:
        bad = ["policy: interval=1", "policy: interval=63", "policy: interval=2"]
    elif p["sub"] == 2:
        bad = ["policy: band=0", "policy: band=0.5", "policy: band=0.01"]
    else:
        bad = ["band(gamma,S,tau) = 0", "band(gamma,S,tau) = 0.5", "band(gamma,S,tau) = 0.01"]
    return good, bad


# ------------------------------------------------------------------------------ selftest


def _check_simulator():
    """Does the simulator behave like the physics says it should? Run before trusting any score."""
    ok = True
    print("simulator validation (a wrong simulator makes every score meaningless)")

    # 1. no costs, daily hedge: the P&L must average zero
    prm = default_params(c=0.0)
    ps = PathSet(prm, 20000, 5)
    s = ps.run(1, 0.0)
    se = s["risk"] / np.sqrt(ps.P)
    good = abs(s["pnl_mean"]) < 3 * se
    ok &= good
    print(f"  c = 0, daily hedge: mean P&L {s['pnl_mean']:+.4f} (3 std errors = {3 * se:.4f})  "
          f"{'ok' if good else 'FAIL'}")

    # 2. hedging error shrinks like 1/sqrt(steps): 63 vs 252 steps should differ by about 2x
    r63 = PathSet(default_params(c=0.0, steps=63), 20000, 5).run(1, 0.0)["risk"]
    r252 = s["risk"]
    good = 1.8 <= r63 / r252 <= 2.2
    ok &= good
    print(f"  c = 0: std at 63 steps / std at 252 steps = {r63 / r252:.2f} (theory 2.00)  "
          f"{'ok' if good else 'FAIL'}")

    # 3. total cost is linear in c for a fixed policy
    a = PathSet(default_params(c=0.005), 4000, 5).run(1, 0.0)["cost"]
    b = PathSet(default_params(c=0.010), 4000, 5).run(1, 0.0)["cost"]
    good = abs(b / a - 2.0) < 1e-9
    ok &= good
    print(f"  cost at c = 0.010 / cost at c = 0.005 = {b / a:.6f} (must be 2)  "
          f"{'ok' if good else 'FAIL'}")

    # 4. never hedging leaves big risk; hedging reduces it hugely
    unh = ps.unhedged()["risk"]
    good = unh > 10 * r252
    ok &= good
    print(f"  unhedged std {unh:.2f} vs daily-hedged std {r252:.2f}  {'ok' if good else 'FAIL'}")

    # 5. determinism, and train/test really are different paths
    j1 = PathSet(default_params(), 2000, 7).run(3, 0.0)["J"]
    j2 = PathSet(default_params(), 2000, 7).run(3, 0.0)["J"]
    j3 = PathSet(default_params(), 2000, 8).run(3, 0.0)["J"]
    good = j1 == j2 and j1 != j3
    ok &= good
    print(f"  same seed -> same J, other seed -> different J  {'ok' if good else 'FAIL'}")

    # 6. higher cost -> wider optimal band (the textbook result)
    bs = []
    for c in (0.005, 0.01):
        pp = PathSet(default_params(c=c), 10000, 5)
        bs.append(best_policy(pp, "band")[1]["band"])
    good = bs[0] < bs[1]
    ok &= good
    print(f"  optimal band at c = 0.005: {bs[0]:.3f}  <  at c = 0.010: {bs[1]:.3f}  "
          f"{'ok' if good else 'FAIL'}")
    return ok


def selftest():
    print(f"level {LEVEL}: {TITLE}\n")
    rc = 0
    if not _check_simulator():
        rc = 1
    print()
    for sub in SUBS:
        p = make(sub)
        ref = reference(p)
        prm = p["prm"]
        gap = ref["naive"]["J"] / ref["Jstar"] - 1
        print(f"{p['name']}: c={prm['c']:g} lam={prm['lam']:g} sigma={prm['sigma']:g} "
              f"T={prm['T']:g} r={prm['r']:g}")
        print(f"   unhedged J {ref['unhedged']['J']:.3f} | daily full hedge J {ref['naive']['J']:.3f} "
              f"| best J {ref['Jstar']:.3f} at {ref['policy']}  "
              f"(daily hedge is {gap * 100:.0f}% worse than best)")
        # Difficulty gate follows the actual loose scoring threshold.
        tolerance = 0.05 if sub == 3 else 0.10
        if gap <= tolerance:
            print(f"   FAIL: daily hedging is within the {tolerance:.0%} reference tolerance")
            rc = 1
        good, bad = offline_candidates(p)
        cases = [("best policy", good[0], (1.0, 1.0)),
                 ("daily full hedge", bad[0] if sub != 3 else "band(gamma,S,tau) = 0", (0.4, 0.4)),
                 ("trade almost never", bad[1], (0.2, 0.4)),
                 ("no answer line", "I would hedge sometimes.", (0.0, 0.0))]
        if sub == 1:
            cases.append(("invalid interval", "policy: interval=0", (0.0, 0.0)))
        if sub == 2:
            cases.append(("band out of range", "policy: band=0.9", (0.0, 0.0)))
        if sub == 3:
            cases += [("log of a negative", "band(gamma,S,tau) = log(-1)", (0.0, 0.0)),
                      ("unsafe code", "band = __import__('os').getcwd()", (0.0, 0.0))]
        for label, ans, want in cases:
            r = check(p, ans)
            okc = want[0] <= r["reward"] <= want[1]
            rc |= 0 if okc else 1
            print(f"   {label:<20} reward {r['reward']:.1f}  {'ok' if okc else f'FAIL (expected {want})'}")
        print()

    # 5.3: structure should beat the best constant band, otherwise the sub-level teaches nothing
    p3 = make(3)
    ps = test_set(p3)
    const = best_policy(ps, "band")
    form = reference(p3)
    gain = const[0] / form["Jstar"] - 1
    print(f"5.3: best constant band J {const[0]:.3f} vs best state-dependent J {form['Jstar']:.3f} "
          f"({gain * 100:.1f}% better)  {'ok' if gain >= 0 else 'FAIL'}")
    rc |= 0 if gain >= 0 else 1

    # the feedback must point the right way
    p2 = make(2)
    r_hi = check(p2, "policy: band=0")                       # trades constantly
    r_lo = check(p2, "policy: band=0.5")                     # barely trades
    d_hi = "Trade less" in r_hi["feedback"]
    d_lo = "Hedge more" in r_lo["feedback"]
    print(f"feedback direction: band=0 -> 'trade less': {'yes' if d_hi else 'NO'};  "
          f"band=0.5 -> 'hedge more': {'yes' if d_lo else 'NO'}")
    rc |= 0 if (d_hi and d_lo) else 1
    print("\nSELFTEST " + ("PASSED" if rc == 0 else "FAILED"))
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", type=int, metavar="SUB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--answer", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.show:
        p = make(a.show, a.seed)
        print(prompt_of(p))
        if a.answer:
            print("\nbest found:", offline_candidates(p)[0][0])
        return
    print(f"level {LEVEL}: {TITLE}")
    for sub in SUBS:
        p = make(sub, a.seed)
        prm = p["prm"]
        print(f"  {sub}. choose {FAMILY[sub]}   (c={prm['c']:g}, lam={prm['lam']:g}, "
              f"sigma={prm['sigma']:g}, T={prm['T']:g})")


if __name__ == "__main__":
    main()
