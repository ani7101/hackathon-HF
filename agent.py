#!/usr/bin/env python3
"""Level-5 agent: propose hedge policies, simulate, grade, and refine.

Run: python agent.py --all --offline
Live: OPT_BASE_URL=http://localhost:8000/v1 python agent.py --all
Offline mode exercises plumbing only; it does not measure model performance.
"""

import argparse
import json
import os
import random
import sys
import time

import level5_hedge

LEVELS = {5: level5_hedge}


# ------------------------------------------------------------------------------ the model


def ask_once(a, prompt):
    import httpx
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=a.max_tokens, temperature=0.6, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body, timeout=900, verify=True)
    if r.status_code != 200:
        raise SystemExit(f"the server returned HTTP {r.status_code}:\n{r.text[:800]}")
    return r.json()["choices"][0]["message"].get("content") or ""


def tool_for(problem):
    """(instructions, requests_in, answer_block, final) for this level's tool."""
    mod = level5_hedge
    return (mod.instructions(problem), mod.requests_in,
            lambda asks: mod.answer_block(problem, asks), mod.FINAL)


def one_attempt(a, problem, prompt, rnd, sample=0):
    """One sample, including its tool exchanges. The model decides what to try; the tool
    answers exactly what was asked and nothing more."""
    if a.offline:
        return fake_model(problem, rnd, sample), 0
    instructions, requests_in, answer_block, final = tool_for(problem)
    convo = prompt if a.no_tools else f"{prompt}\n\n{instructions}"
    used = 0
    for step in range(a.tool_steps + 1):
        reply = ask_once(a, convo)
        asks = [] if a.no_tools else requests_in(reply)
        if not asks or step == a.tool_steps:
            return reply, used
        used += len(asks)
        convo = f"{convo}\n\n{reply}\n\n{answer_block(asks)}\n\n{final}"
    return reply, used


def ask_round(a, problem, prompt, rnd):
    import concurrent.futures as cf
    if a.offline:
        # Warm the shared reference once before starting sample threads.
        level5_hedge.reference(problem)
    with cf.ThreadPoolExecutor(max_workers=a.samples) as ex:
        futures = [ex.submit(one_attempt, a, problem, prompt, rnd, sample)
                   for sample in range(a.samples)]
        return [f.result() for f in futures]


def fake_model(problem, rnd, sample=0):
    """Offline stand-in so the loop can be exercised with no model. It improves each round.
    Never report a number that came from here."""
    rng = random.Random(f"{problem['name']}-{problem['seed']}-{rnd}-{sample}")
    good, bad = LEVELS[problem["level"]].offline_candidates(problem)
    p_good = min(0.9, 0.15 + 0.25 * rnd)
    return rng.choice(good) if rng.random() < p_good else rng.choice(bad)


# ------------------------------------------------------------------------------ the loop


def attempt_line(problem, expr):
    return f"band(gamma,S,tau) = {expr}" if problem["sub"] == 3 else f"policy: {expr}"


def next_prompt(a, problem, base_prompt, best):
    line = attempt_line(problem, best["expr"])
    if a.feedback == "none":
        return base_prompt
    if a.feedback == "score":
        return (f"{base_prompt}\n\nA previous attempt was:\n  {line}\n"
                f"It scored {best['reward']:.1f} out of 1.0. Improve it.")
    return (f"{base_prompt}\n\nA previous attempt was:\n  {line}\n"
            f"A checker found this problem with it: {best['feedback']}\nFix it.")


def solve(problem, a, log):
    mod = LEVELS[problem["level"]]
    problem["hint"] = a.hint
    base_prompt = mod.prompt_of(problem)
    print(f"\n=========== {problem['name']} ===========")
    print(base_prompt)
    prompt, best_ever = base_prompt, 0.0
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        attempts = ask_round(a, problem, prompt, rnd)
        graded = [mod.check(problem, ans) for ans, _ in attempts]
        for (ans, used), g in zip(attempts, graded):
            log.write(json.dumps(dict(
                tag=a.tag, problem=problem["name"], seed=problem["seed"], round=rnd,
                no_tools=a.no_tools, feedback=a.feedback, hint=a.hint, offline=a.offline,
                prompt=prompt, answer=ans, tool_calls=used, reward=g["reward"],
                parts=g["parts"])) + "\n")
        log.flush()
        rewards = [g["reward"] for g in graded]
        best = max(graded, key=lambda g: g["reward"])
        best_ever = max(best_ever, best["reward"])
        tool_calls = sum(u for _, u in attempts)
        print(f"\nround {rnd}: rewards {rewards}  mean {sum(rewards) / len(rewards):.2f}  "
              f"best {best['reward']:.1f}  {tool_calls} tool call(s)  "
              f"({time.perf_counter() - t0:.1f}s)")
        if best["reward"] == 1.0:
            print(f"SOLVED: {attempt_line(problem, best['expr'])[:300]}")
            return 1.0, rnd + 1
        print(f"  best: {best['expr'][:200]}")
        print(f"  checker: {best['feedback']}")
        prompt = next_prompt(a, problem, base_prompt, best)
    print(f"not solved in {a.rounds} rounds; best reward {best_ever:.1f}")
    return best_ever, a.rounds


def main():
    env = os.environ.get
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=5, choices=sorted(LEVELS))
    ap.add_argument("--sub", type=int)
    ap.add_argument("--all", action="store_true", help="every sub-problem in this level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=int(env("OPT_MAX_TOKENS", 1500)))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--tool-steps", type=int, default=1,
                    help="rounds of tool exchanges allowed per attempt")
    ap.add_argument("--no-tools", action="store_true", help="withhold the SIM tool")
    ap.add_argument("--feedback", choices=("full", "score", "none"), default="full")
    ap.add_argument("--hint", type=int, default=1, choices=(0, 1, 2))
    ap.add_argument("--model", default=env("OPT_MODEL", env("HEATROD_MODEL", "Qwen/Qwen3-8B")))
    ap.add_argument("--base", default=env("OPT_BASE_URL", env("HEATROD_BASE_URL")))
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--tag", default="", help="free-text label stored with every logged attempt")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    if min(a.samples, a.rounds, a.max_tokens) < 1 or a.tool_steps < 0:
        ap.error("samples, rounds and max-tokens must be positive; tool-steps must be nonnegative")
    if a.sub is not None and a.sub not in level5_hedge.SUBS:
        ap.error("sub must be 1, 2 or 3")

    if not a.base and not a.offline:
        sys.exit("set OPT_BASE_URL (or HEATROD_BASE_URL) to your vLLM endpoint, or pass --offline")
    if a.offline:
        print("*** OFFLINE: fake generator, numbers are meaningless ***")

    mod = LEVELS[a.level]
    subs = mod.SUBS if (a.all or not a.sub) else (a.sub,)
    results = []
    with open(a.log, "a") as log:
        for sub in subs:
            reward, rounds = solve(mod.make(sub, a.seed), a, log)
            results.append((f"level{a.level}.{sub}", reward, rounds))

    print(f"\n=========== summary: level {a.level} ===========")
    for name, reward, rounds in results:
        print(f"  {name:<12} reward {reward:.1f} after {rounds} round(s)"
              + ("  SOLVED" if reward == 1.0 else ""))
    print(f"  solved {sum(1 for _, r, _ in results if r == 1.0)}/{len(results)}")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
