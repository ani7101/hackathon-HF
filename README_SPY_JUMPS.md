# SPY AI hedging with random jumps

`portfolio_agent_realworld.py` is a separate copy of the AI runner. Copy it,
`market_jumps.py`, and `spy_calibration.py` to the existing project on your pod.
It still needs the shared `market.py`, `portfolio.py`, `risk.py`, and `search.py`.

```bash
python portfolio_agent_realworld.py --base http://localhost:8000/v1 \
  --calibrate-spy --iterations 15 --paths 2000 --test-paths 2000 \
  --capital 1000 --drawdown-limit 0.10 --loss-limit 100 --cash-floor 0 \
  --output artifacts/portfolio-spy-jumps.json
```

The live calibration fetches five years of daily SPY data from Yahoo Finance.
If access fails, use `--spy-csv spy.csv` instead of `--calibrate-spy`. Required CSV
columns: `Date` (YYYY-MM-DD), `Close`, and `Adj Close`; at least 253 daily rows.
The latest Close sets the starting price and at-the-money strike. Returns from
adjusted closes estimate diffusion and jump proxies; diffusion volatility also
sets fixed marking volatility. Review the data dates in the output.

## Market generation

For each time interval dt, N ~ Poisson(lambda * dt). Each log jump is normal
with mean m and standard deviation s; its price multiplier is exp(log jump).
With N events, total log jump is N*m + sqrt(N)*s*Z. This permits zero, one or
multiple shocks on any day. No fixed step 63 event exists.

```
kappa = exp(m + s*s/2) - 1
log return = (mu - lambda*kappa - sigma*sigma/2)*dt
             + sigma*sqrt(dt)*Z + total_log_jump
```

mu is a physical-measure expected price-return rate, distinct from the cash rate.
The compensator preserves E[S(T)] = S(0)*exp(mu*T). Base uses the fitted values.
High volatility multiplies diffusion volatility by 1.75. The optional mild downside/upside stress groups preserve fitted jump intensity
and shift log jump means by -0.01 and +0.01. Jump standard deviation is unchanged.
Use `--stress-profile severe` to reproduce the old doubled frequency and -0.08/+0.12
mean shifts. Use `--scenario-set base` to omit all added stress families while
retaining fitted synthetic jumps. The high-volatility multiplier remains 1.75.
Those stress transformations are assumptions, not calibrated probabilities.

## Calibration limitations and defaults

Extreme daily returns are identified above 3 robust standard deviations from
the median. The robust deviation is median absolute deviation / 0.67448975.
Their frequency and residual sizes proxy jump intensity and log jump sizes.
Ordinary-day volatility estimates diffusion. This is a transparent heuristic,
not a structural separation of continuous shocks and true jumps. Daily extremes
contain both, and threshold choice affects the estimates. The drift uses a
moment match to the full sample mean log return and fitted jump contribution.

Without calibration flags, all market values are illustrative: price/strike 100,
annual expected return 6%, diffusion volatility 20%, two jumps/year, log jump mean
-0.04, log jump std 0.08. They are NOT fitted SPY values.

Cash rate 4%, fees 0.05% (5 basis points of trade value), investor capital 1000,
and risk limits remain explicit assumptions even with historical calibration.
The obligation is one call on one share, not a standard 100-share option contract.
Adjusted returns proxy total returns; dividends are not explicitly paid in the
portfolio. Marks and hedge deltas remain Black-Scholes with fixed volatility;
they do not use jump-aware pricing. Poisson intensity is constant within a group:
no clustered shocks, stochastic volatility or margin model is included.

Outputs record calibration source/date range, fitted parameters, scenario seeds,
observed jump counts, candidate metrics and fresh-test results. Historical fitting
does not guarantee feasibility or real-market performance. Fresh simulated paths
test sampling robustness under the same assumed model, not an out-of-time backtest.

Reference: Merton (1976), https://doi.org/10.1016/0304-405X(76)90022-2

`observed_jumps` counts events in generated simulations, not in source data.
`event_source` and `induced_stress` identify this explicitly in output diagnostics.
Reducing added stresses makes a test easier; it does not make a policy safer.

## Simple comparison (new default)

`--stress-profile simple` tests two groups: fitted base and `more_jumps`.
The second group has 1.25 times the fitted frequency with identical jump-size
mean and standard deviation. It adds no high-volatility group or mean shifts.
With the previously supplied SPY fit this is 8.44 versus 10.55 expected events/year.
Both are synthetic. Extra frequency is an explicit stress assumption.

Run a baseline using a saved AI result to preserve calibration and capital:

```bash
python baseline_spy.py --report artifacts/portfolio-spy-simple.json
```

This compares daily delta rebalancing, a fixed 0.10-share band, and a grid of 50
unique (band,response) policies. All use identical search paths and independent
fresh paths. Compare AI and grid at equal unique simulator evaluation counts;
50 model proposals may contain invalid or duplicate candidates. Existing runs
with different stresses cannot be directly compared to a new simple run.
