"""Execute a band policy and return full equity/cash trajectories.

One short cash-settled call on one share, positive investor capital, proportional
fees, and symmetric cash interest. Stocks are liquidated at expiry. Breaches are
recorded for rejection by the evaluator; forced liquidation is not modeled.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Policy:
    base_band: float
    drawdown_response: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.base_band) or not 0 <= self.base_band <= 0.5:
            raise ValueError('base_band must be finite and in [0, 0.5]')
        if not np.isfinite(self.drawdown_response) or self.drawdown_response < 0:
            raise ValueError('drawdown_response must be finite and nonnegative')


def simulate(policy, scenarios):
    """Return arrays equity/cash (paths, 1 + 2*(steps+1)), fees (paths,), and times.

Each decision contributes a pre-trade and post-trade observation. Repeated times
are intentional: transaction fees must not hide an immediately preceding peak.
"""
    config = scenarios.config
    prices, marks, deltas = scenarios.prices, scenarios.option_values, scenarios.deltas
    paths, columns = prices.shape
    dt = config.maturity / config.steps
    cash = np.full(paths, config.capital) + marks[:, 0]
    shares = np.zeros(paths)
    total_fees = np.zeros(paths)
    peak = np.full(paths, config.capital)
    equity_history = [peak.copy()]
    cash_history = [cash.copy()]
    times = [0.0]

    for i in range(columns):
        if i > 0:
            cash *= np.exp(config.rate * dt)
        spot = prices[:, i]
        cash_before = cash.copy()
        equity_before = cash + shares * spot - marks[:, i]
        peak = np.maximum(peak, equity_before)
        drawdown = (peak - equity_before) / peak
        if i == 0:
            trade = deltas[:, i] - shares
        elif i == columns - 1:
            trade = -shares
        else:
            band = np.clip(policy.base_band / (1 + policy.drawdown_response * drawdown), 0, 0.5)
            gap = deltas[:, i] - shares
            trade = np.where(np.abs(gap) > band, gap, 0.0)
        fee = config.fee_rate * spot * np.abs(trade)
        cash -= trade * spot + fee
        shares += trade
        total_fees += fee
        if i == columns - 1:
            cash -= marks[:, i]
            equity_after = cash.copy()
        else:
            equity_after = cash + shares * spot - marks[:, i]
        equity_history.extend([equity_before, equity_after])
        cash_history.extend([cash_before, cash.copy()])
        times.extend([i * dt, i * dt])

    return {
        'equity': np.stack(equity_history, axis=1),
        'cash': np.stack(cash_history, axis=1),
        'times': np.asarray(times),
        'fees': total_fees,
        'initial_capital': config.capital,
        'terminal_target': config.capital * np.exp(config.rate * config.maturity),
    }
