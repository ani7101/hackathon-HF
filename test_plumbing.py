#!/usr/bin/env python3
"""
test_plumbing.py -- does the whole loop work, tools included, with NO model server?

A scripted "model" replies in the exact shape a real one would: first it asks for tool calls
(SIM:), then, after the results come back in the prompt, it gives a final answer.
This proves the parts a real run depends on: the request is parsed, the tool runs, its output
reaches the next prompt, the final answer is graded, and the attempt is logged.

    python test_plumbing.py
"""

import argparse
import io
import json
import os
import sys
import tempfile

import agent
import level5_hedge


def run(level, sub, first_reply, final_reply, must_appear):
    seen = []

    def scripted(a, prompt):
        seen.append(prompt)
        return first_reply if len(seen) == 1 else final_reply

    agent.ask_once = scripted
    a = argparse.Namespace(offline=False, no_tools=False, tool_steps=1, samples=1, rounds=1,
                           feedback="full", hint=1, think=False, tag="plumbing")
    mod = agent.LEVELS[level]
    problem = mod.make(sub)
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as log:
        so = sys.stdout
        sys.stdout = io.StringIO()
        try:
            reward, rounds = agent.solve(problem, a, log)
        finally:
            sys.stdout = so
        log.flush()
        logged = [json.loads(l) for l in open(log.name)]
    os.unlink(log.name)
    ok = (reward == 1.0
          and len(seen) == 2                                  # one tool exchange, then the answer
          and all(s in seen[1] for s in must_appear)          # tool output reached the next prompt
          and len(logged) == 1 and logged[0]["tool_calls"] >= 1 and logged[0]["reward"] == 1.0)
    return ok, reward, seen


def main():
    bad = 0

    p5 = level5_hedge.make(2)
    best = level5_hedge.offline_candidates(p5)[0][0]
    ok, reward, seen = run(
        5, 2,
        "Let me try two bands.\nSIM: band=0.05\nSIM: band=0.12",
        "The second looked better, but trade-offs matter.\n" + best,
        ["practice paths", "J ="])
    print(f"level 5 tool loop (SIM: -> answer -> graded -> logged): reward {reward}  "
          f"{'ok' if ok else 'FAIL'}")
    bad += not ok

    # the tools must refuse to do anything but what they advertise
    r = level5_hedge.sim(level5_hedge.make(2), "band=7")
    refused = r.startswith("rejected")
    print(f"simulator refuses an out-of-range band: {'ok' if refused else 'FAIL'}")
    bad += not refused

    print("\nPLUMBING " + ("PASSED" if not bad else "FAILED"))
    return int(bool(bad))


if __name__ == "__main__":
    sys.exit(main())
