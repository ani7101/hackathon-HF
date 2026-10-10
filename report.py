#!/usr/bin/env python3
"""
report.py -- summarize attempts.jsonl: how many problems were solved, in how many rounds, per
configuration. One run is not a result, so this groups by --tag and counts runs.

    python report.py attempts.jsonl
    python report.py attempts.jsonl --tag baseline --tag no-tools

Offline runs are excluded (their numbers are meaningless) unless you pass --include-offline.
"""

import argparse
import collections
import json


def load(path, include_offline):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if r.get("offline") and not include_offline:
            continue
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--tag", action="append")
    ap.add_argument("--include-offline", action="store_true")
    a = ap.parse_args()

    rows = load(a.log, a.include_offline)
    if a.tag:
        rows = [r for r in rows if r.get("tag") in a.tag]
    if not rows:
        print("no attempts to report (offline runs are excluded by default)")
        return

    # one "run" = (tag, problem, seed); its first-solved round is the earliest round with reward 1.0
    runs = collections.defaultdict(dict)
    for r in rows:
        key = (r.get("tag", ""), r["problem"], r["seed"])
        best = runs[key].setdefault(r["round"], 0.0)
        runs[key][r["round"]] = max(best, r["reward"])

    table = collections.defaultdict(list)
    for (tag, problem, seed), rounds in runs.items():
        solved_at = next((rd + 1 for rd in sorted(rounds) if rounds[rd] >= 0.9999), None)
        table[(tag, problem)].append((solved_at, max(rounds.values())))

    print(f"{'tag':<14} {'problem':<10} {'runs':>4} {'solved':>7} {'median round':>13} {'best reward (mean)':>19}")
    for (tag, problem), res in sorted(table.items()):
        n = len(res)
        solved = [s for s, _ in res if s is not None]
        med = sorted(solved)[len(solved) // 2] if solved else "-"
        print(f"{tag or '-':<14} {problem:<10} {n:>4} {len(solved):>3}/{n:<3} {str(med):>13} "
              f"{sum(b for _, b in res) / n:>19.2f}")

    tc = [r.get("tool_calls", 0) for r in rows]
    print(f"\n{len(rows)} attempts logged; {sum(tc)} tool calls in total "
          f"({sum(1 for t in tc if t) / len(tc) * 100:.0f}% of attempts used a tool)")


if __name__ == "__main__":
    main()
