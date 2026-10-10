> Historical audit before the level-5-only change. The missing-core blocker is now resolved for level 5, level 3 is removed, reference ranges and thresholds are corrected, and TLS verification is enabled. See README.md for current setup. Earlier results used a smaller reference grid.

# Pricing agent audit — 2026-10-10

## Setup and execution status

A local environment exists at `../.venv`, using the installed system packages. Activate it with `source ../.venv/bin/activate` from this folder. Exact relevant versions are in `artifacts/local-environment.txt`. `run_all.sh` now has executable permission.

The original agent was attempted and fails at import: `ModuleNotFoundError: No module named 'level3_barrier'`. Both `level3_barrier.py` and `optcheck.py` are absent. They are also absent from the checked-out Trainium repository's tracked files. This blocks both levels, original selftests, plumbing tests, and offline mode. Missing originals should be restored; any reconstruction must be identified and independently validated as a replacement.

No model is listening at localhost:8000. The configured workshop Kubernetes cluster is reachable outside the local network sandbox and has running seat pods. The user's assigned seat is still needed before entering or changing a pod. Credentials were not printed or copied into artifacts.

`diagnostics/simulator_audit.py` is an explicitly isolated diagnostic: it supplies only a temporary in-memory Black–Scholes/normal-CDF adapter for the absent shared module, then runs the existing `hedgesim.py`. It does not restore either missing file, exercise formula parsing, or run an LLM. Run from this directory with:

```bash
../.venv/bin/python diagnostics/simulator_audit.py
```

## What the code actually does

The shape is inherited from `trainium-agent-labs/projects/01-heat-rod-pde/agent.py`:

1. Generate a seeded problem.
2. Ask one OpenAI-compatible model for four independently sampled answers concurrently (default temperature 0.6).
3. Allow each answer one tool exchange by default. Level 3 requests a calculator with `COMPUTE:`; level 5 requests practice-path simulations with `SIM:`.
4. Grade all answers with a deterministic numerical checker.
5. Feed the current round's highest-scoring answer and feedback into the next prompt, up to four rounds by default. Stop early at reward 1.0.
6. Append attempts to JSONL. There is no optimizer, backpropagation, weight update, or RL training in this code.

There are not currently two independent competing agents. Concurrent samples use the same model and prompt. The checker and simulator run locally on CPU; Trainium hosts model inference.

Level 3 is intended to synthesize barrier-option formulas; its actual checker cannot be inspected because its implementation is missing. Level 5 does not generate a new option price: it receives a Black–Scholes call premium and optimizes a hedge policy. It minimizes simulated transaction cost plus lambda times terminal P&L standard deviation, under risk-neutral geometric Brownian motion. Its choices are a rebalance interval, a constant no-trade band, or a state-dependent band formula. Initial delta setup and final settlement incur transaction costs.

## Measured diagnostic baseline

Parameters: S0=K=100, T=1, r=0, sigma=0.2, c=0.005, lambda=1, 252 steps. These are simulator defaults, not all generated level-5 problems.

Policies were selected on 2,000 practice paths with seed 101, then evaluated without retuning on 20,000 separate paths with seed 9001.

| Policy | Practice J | Separate-path J |
|---|---:|---:|
| Daily delta hedge | 3.873732 | 3.879448 |
| Best practice interval: 7 | 2.400647 | 2.481161 |
| Best practice constant band: 0.095 | 2.313383 | 2.354366 |
| Best practice structured band: kappa=1.7 | 2.207319 | 2.257150 |

Structured band: `clip(1.7 * (c*S*gamma**2)**(1/3), 0, 1)`. Raw results are in `artifacts/simulator-audit.json` and `artifacts/simulator-validation.json`. This is one seeded simulation comparison, not evidence of model improvement or a true-market optimum. All six existing simulator-only checks in `level5_hedge._check_simulator()` passed using the temporary numerical adapter; output is in `artifacts/simulator-checks.txt`. Missing parsing functions were replaced with fail-fast placeholders for that isolated import and were never called. Neither the full original checker selftests nor the original agent have passed.

## Issues to resolve before an optimization claim

- Missing source modules prevent the actual baseline from running.
- Level 5.3 README promises 5%/1% thresholds, but `check()` uses 10%/3% for every sublevel. It has no structural requirement that prevents a constant formula receiving full reward.
- Validation paths are fixed and their scores are returned as iterative feedback. Keep a third, untouched final evaluation set; existing feedback paths are not a final holdout.
- The reference is a limited grid, not a proven optimum. Intervals search only 17 values although the prompt permits integers 1–63; constant bands search 0–0.30 although the prompt permits 0–0.50; arbitrary formulas are benchmarked against only constants and a one-parameter family.
- Reward is quantized in 0.2 steps. Equal reward ties choose the first answer, even if another has better J. Preserve continuous objective and a best-ever candidate alongside the exploration history.
- The next prompt retains only one current-round answer. Prior failures and a better earlier candidate can be forgotten. Add bounded failure memory and duplicate detection.
- Logging lacks run UUID, model/settings, actual tool transcript, checker feedback, continuous J, and timings. `report.py` merges repeated runs sharing tag/problem/seed, so independent repetitions can be conflated.
- `--seed` controls problem generation, not the model's sampling. Offline sampling also uses a shared counter across threads. Reproducibility requires explicit per-sample identities and supported server sampling seeds.
- `ask_once()` disables TLS verification. Make verification the default and any workshop-specific exception explicit.
- Formula-family grid evaluation recomputes the same cube-root base for every kappa. Cache this per path set, cache repeated policy evaluations, and benchmark before changing the math.

## Two-agent design and the meaning of convergence

Decide the target first: a numerical price, the best hedge policy, and a robust adversarial market game are different objectives. Adding an adversarial market agent changes the current fixed-model mean/std optimization into a minimax problem.

A useful architecture for the existing hedge objective is a proposer that tries to improve the incumbent policy and a challenger that seeks better competing policies or counterexamples. A deterministic referee controls eligibility, evaluations, and bounds. Neither agent is allowed to certify its own proposal. Freeze the objective, parameter domain, paths and evaluation budget for each run.

For an actual certified price bracket, the desired invariant is `L_t <= V* <= U_t`. The lower-side agent tries to raise L; the upper-side agent tries to lower U. Accept updates only with valid certificates. If a fallback solver can ALWAYS preserve containment and enforce `U_(t+1)-L_(t+1) <= rho*(U_t-L_t)` for a fixed `0 < rho < 1`, then after N iterations width is at most `rho**N * (U_0-L_0)`. A midpoint estimate has error at most half that width. To achieve bracket width epsilon, require `N >= ceil(log((U_0-L_0)/epsilon) / log(1/rho))`, or zero rounds if the initial width already suffices.

This is a conditional theorem, not a capability the repository already has. A reliable sign oracle for a monotone root admits bisection with rho=1/2. Two language models agreeing on prices does not provide that oracle. No-arbitrage bounds alone do not supply geometric contraction. Monte Carlo estimates give probabilistic uncertainty, not unconditional exact certificates; adaptively repeated confidence intervals also need simultaneous coverage control.

For the existing hedge simulator, do not shrink a continuous parameter interval around the winning sample and claim global convergence. Trading thresholds make the empirical objective non-smooth; the code establishes neither convexity nor unimodality. Unsampled regions may contain better policies.

The simplest honest finite-N guarantee is initially a fixed finite policy set C with M candidates. Let the two agents prioritize distinct candidates; the controller fills missed/duplicate slots deterministically. Evaluate at least ceil(M/N) new candidates per round until all are covered. After at most N rounds the incumbent is the exact minimum over C on the frozen empirical paths (up to numerical precision). This guarantees the finite-grid objective, not the continuous/global or expected real-world optimum. It needs M total evaluations in the worst case and is an audit baseline rather than a claimed speedup. A shrinking set of untested candidates is valid; shrinking an unchecked spatial interval is not.

For a true minimax extension, define a finite market-scenario set as well as policies and specify whether mixed policies are allowed. Policy and adversary agents can propose best responses, while a solver maintains primal/dual bounds. Add a full finite-game solve fallback at the budget cap before claiming an epsilon gap. Unrestricted neural agents cannot guarantee exact equilibrium after arbitrary N iterations.

## Next execution steps

Restore or explicitly reconstruct and validate the two missing modules. Identify the assigned seat/model endpoint, verify the checker selftests, run a small real baseline, then repeat across seeds. Record continuous J, total model/tool calls and wall time. Only then compare a two-agent implementation against both the current single-model loop and a deterministic search at matched evaluation budgets.
