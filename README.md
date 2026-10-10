# Level 5: hedging with transaction costs

For a detailed explanation of every module, the execution flow, simulator math, scoring and extension points, see [the code walkthrough](README_CODE_WALKTHROUGH.md).

This project optimizes the hedge of a short European call. Level 3 and its calculator have been removed. There is no dependency on `optcheck.py` or `level3_barrier.py`.

The agent proposes policies, tries them with the `SIM:` tool, receives numerical feedback, and refines its next answer. Four candidates are sampled per round by default, for up to four rounds. This is inference and search, not model training.

## Objective and choices

Minimize `J = mean(transaction cost) + lambda * std(terminal P&L)`.

| Subproblem | Choice |
|---|---|
| 5.1 | Rebalance every m steps, integer m in 1–63 |
| 5.2 | Trade when the delta gap exceeds b, b in 0–0.5 |
| 5.3 | Supply a band formula depending on gamma, S and time remaining |

Trading is discrete and incurs proportional costs at initial setup, rebalancing and settlement. The model can request up to four practice simulations per tool exchange.

**Model assumptions:** the current benchmark still simulates constant-volatility geometric Brownian motion and uses Black–Scholes for the initial premium and target delta/gamma. Removing formula-pricing level 3 does not remove those assumptions. Results measure hedging under this simulated market, not realistic market pricing. J measures cost and dispersion; it does not penalize mean hedging loss separately. Before introducing jumps, volatility mismatch or adversarial market dynamics, specify the intended risk objective and keep a baseline for comparison.

## Setup and run

From this folder, use the prepared environment or create a fresh one:

```bash
python3 -m venv ../.venv
../.venv/bin/python -m pip install -r requirements.txt
./run_all.sh selftest
./run_all.sh offline
```

`selftest` checks safe formula evaluation, the simulator, scoring and the scripted tool-to-feedback loop. Its problem-difficulty check uses the scoring tolerance (10%, or 5% for 5.3), replacing the old unrelated 15% cutoff that failed on the supplied 5.2 problem. `offline` uses deliberately scripted candidates and must not be reported as model performance.

For a live OpenAI-compatible endpoint:

```bash
export OPT_BASE_URL=http://localhost:8000/v1
export OPT_MODEL=Qwen/Qwen3-8B
../.venv/bin/python agent.py --sub 2 --samples 4 --rounds 4
./run_all.sh real
./run_all.sh experiments
../.venv/bin/python report.py artifacts/attempts.jsonl
```

`--level 5` remains accepted for compatibility and is the only supported level. `real` runs three seeds sequentially and propagates failures. `experiments` compares no-tools, score-only feedback and no hint. Override `PYTHON` to select a different interpreter. HTTPS certificate verification is enabled.

## Checker and limits

Each of five checks contributes 0.2: valid policy, beats unhedged, beats daily hedge, within the loose reference tolerance, and within the tight tolerance. Tolerances are 10%/3% for 5.1–5.2 and 5%/1% for 5.3. Constants are allowed in 5.3 and can receive full credit if their actual score meets the thresholds; there is no artificial structural penalty.

The reference searches all 63 intervals, bands 0–0.5 in steps of 0.005, and (for 5.3) constants plus 80 coefficients of the family `kappa*(c*S*gamma**2)**(1/3)`. This is a finite grid benchmark, not a guaranteed continuous or arbitrary-formula optimum. The cube-root base is cached across coefficients.

Practice uses 2,000 paths (seed 101); checker feedback uses 20,000 fixed validation paths (seed 9001). Repeated feedback makes these validation data, not an untouched final test. Use fresh seeds for final evaluations and multiple model runs before claiming improvement.

Two competing agents and certified contraction bounds are not implemented yet. A shrinking search interval alone cannot guarantee a global optimum of this threshold-based, non-smooth objective. The earlier design discussion is preserved in `AUDIT.md`; its original missing-module blockers are now resolved for level 5.

## Files

- `agent.py`: sampling, SIM tool exchange, feedback and attempt logging.
- `level5_hedge.py`: problems, policy parser, checker, reference and simulator selftests.
- `hedgesim.py`: path simulation and hedge execution.
- `hedge_core.py`: numerical benchmark and restricted arithmetic evaluator.
- `test_hedge_core.py`, `test_plumbing.py`: core and end-to-end scripted checks.
- `report.py`: reports logged runs; offline results excluded by default.
- `make_figures.py`: hedge-only figures; needs optional `matplotlib`.
- `diagnostics/simulator_audit.py`: deterministic reference search, no model.

`archive/README-before-level5.pdf` and older audit artifacts are historical and do not describe the current executable project.
