#!/usr/bin/env python3
"""
make_figures.py -- draw the figures used in the PDF from the real code (no illustrative numbers)
and save the numbers the PDF quotes to results/numbers.json.

    python make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hedgesim
import level5_hedge

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

# validated categorical slots 1-3 (validate_palette.js --mode light --pairs all: PASS)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.5,
})
numbers = {}


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------- figure 3
p2 = level5_hedge.make(2)
ps = level5_hedge.test_set(p2)
prm = p2["prm"]
bands = np.array(hedgesim.BANDS)
res = [ps.run(1, float(b)) for b in bands]
cost = np.array([x["cost"] for x in res])
risk = prm["lam"] * np.array([x["risk"] for x in res])
J = np.array([x["J"] for x in res])
unh = ps.unhedged()
kbest = int(np.argmin(J))
fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.plot(bands, cost, color=ORANGE, lw=2)
ax.plot(bands, risk, color=AQUA, lw=2)
ax.plot(bands, J, color=BLUE, lw=2.4)
ax.plot([bands[kbest]], [J[kbest]], "o", color=BLUE, ms=9, mec=SURFACE, mew=2)
ax.plot([0], [J[0]], "o", color=BLUE, ms=9, mec=SURFACE, mew=2)
ax.annotate(f"best band {bands[kbest]:.3f}\nJ = {J[kbest]:.2f}", (bands[kbest], J[kbest]),
            xytext=(bands[kbest] + 0.025, J[kbest] + 2.2), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.annotate(f"hedge every step\nJ = {J[0]:.2f}", (0, J[0]), xytext=(0.012, J[0] + 3.0),
            fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.text(bands[-1] + 0.004, J[-1], "J (total)", color=INK, va="center", fontsize=8.5)
ax.text(bands[-1] + 0.004, risk[-1], "risk term", color=INK, va="center", fontsize=8.5)
ax.text(bands[-1] + 0.004, cost[-1], "trading cost", color=INK, va="center", fontsize=8.5)
ax.set_xlim(0, 0.60)
ax.set_xlabel("no-trade band b (shares of delta)")
ax.set_ylabel("expected cost, or lam x std(P&L)")
ax.set_title("Trading less saves cost but adds risk: the best band is in between")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=BLUE, lw=2.4, label="J = cost + lam x risk"),
                   Line2D([], [], color=ORANGE, lw=2, label="expected trading cost"),
                   Line2D([], [], color=AQUA, lw=2, label="lam x std of P&L")],
          loc="upper right", bbox_to_anchor=(1.0, 0.82))
finish(fig, "fig_hedge_tradeoff.png")
numbers["hedge_example"] = dict(
    c=prm["c"], lam=prm["lam"], sigma=prm["sigma"], T=prm["T"], steps=prm["steps"],
    unhedged_J=unh["J"], naive_J=float(J[0]), naive_cost=float(cost[0]),
    naive_risk=float(res[0]["risk"]), naive_trades=float(res[0]["trades"]),
    best_band=float(bands[kbest]), best_J=float(J[kbest]),
    best_cost=float(cost[kbest]), best_risk=float(res[kbest]["risk"]),
    best_trades=float(res[kbest]["trades"]),
    gap_pct=float((J[0] / J[kbest] - 1) * 100))

# ---------------------------------------------------------------------------- figure 4
iv = [(m, ps.run(m, 0.0)) for m in hedgesim.INTERVALS if m < ps.n]
bd = [(b, ps.run(1, float(b))) for b in hedgesim.BANDS]
fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.plot([x[1]["trades"] for x in iv], [x[1]["J"] for x in iv], "-o", color=ORANGE, lw=2, ms=6,
        mec=SURFACE, mew=1.2)
ax.plot([x[1]["trades"] for x in bd], [x[1]["J"] for x in bd], "-", color=BLUE, lw=2)
ax.set_xscale("log")
ax.set_ylim(top=min(max(x[1]["J"] for x in iv), 12))
xi = [x[1]["trades"] for x in iv]
ax.text(xi[3], iv[3][1]["J"] + 0.55, "fixed interval", color=INK, fontsize=8.5, ha="center")
xb = [x[1]["trades"] for x in bd]
ax.text(xb[22], bd[22][1]["J"] - 0.95, "no-trade band", color=INK, fontsize=8.5, ha="left")
ax.set_xlabel("average number of trades per path (log scale)")
ax.set_ylabel("J (lower is better)")
ax.set_title("For the same number of trades, a band beats a fixed schedule")
ax.legend(handles=[Line2D([], [], color=ORANGE, lw=2, marker="o", ms=6, label="rebalance every m steps"),
                   Line2D([], [], color=BLUE, lw=2, label="rebalance only outside a band")],
          loc="upper right")
finish(fig, "fig_interval_vs_band.png")
numbers["interval_best"] = dict(m=int(min(iv, key=lambda x: x[1]["J"])[0]),
                                J=float(min(x["J"] for _, x in iv)))

# ------------------------------------------------------- reference numbers for each problem
numbers["level5_problems"] = []
for sub in level5_hedge.SUBS:
    p = level5_hedge.make(sub)
    ref = level5_hedge.reference(p)
    numbers["level5_problems"].append(dict(
        name=p["name"], c=p["prm"]["c"], lam=p["prm"]["lam"], sigma=p["prm"]["sigma"],
        T=p["prm"]["T"], r=p["prm"]["r"], unhedged=ref["unhedged"]["J"], naive=ref["naive"]["J"],
        best=ref["Jstar"], policy=ref["policy"],
        gap_pct=(ref["naive"]["J"] / ref["Jstar"] - 1) * 100))
const = hedgesim.best_policy(level5_hedge.test_set(level5_hedge.make(3)), "band")
numbers["const_vs_formula"] = dict(const=const[0], formula=level5_hedge.reference(level5_hedge.make(3))["Jstar"])

with open(os.path.join(OUT, "numbers.json"), "w") as f:
    json.dump(numbers, f, indent=1, default=float)
print("wrote numbers.json")
