"""Market paths and benchmark option marks for the portfolio experiment.

Realized volatility drives the simulated stock. Fixed mark_vol drives the option
valuation and delta. Gap scenarios are stress tests, not calibrated probabilities.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class Config:
    spot: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0
    steps: int = 252
    rate: float = 0.02
    mark_vol: float = 0.20
    fee_rate: float = 0.005
    capital: float = 100.0

    def __post_init__(self):
        for name in ('spot', 'strike', 'maturity', 'mark_vol', 'capital'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(self.rate):
            raise ValueError('rate must be finite')
        if not np.isfinite(self.fee_rate) or self.fee_rate < 0:
            raise ValueError('fee_rate must be finite and nonnegative')
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError('steps must be a positive integer')


@dataclass
class Scenarios:
    config: Config
    prices: np.ndarray
    option_values: np.ndarray
    deltas: np.ndarray

    def __post_init__(self):
        expected_columns = self.config.steps + 1
        if self.prices.ndim != 2 or self.prices.shape[0] < 1:
            raise ValueError('prices must be a nonempty two-dimensional array')
        if self.prices.shape[1] != expected_columns:
            raise ValueError('prices must have steps + 1 columns')
        for array in (self.prices, self.option_values, self.deltas):
            if array.shape != self.prices.shape or not np.isfinite(array).all():
                raise ValueError('scenario arrays must have matching shapes and finite values')
        if np.any(self.prices <= 0) or not np.allclose(self.prices[:, 0], self.config.spot):
            raise ValueError('prices must be positive and start at config.spot')
        if np.any(self.option_values < 0) or np.any((self.deltas < 0) | (self.deltas > 1)):
            raise ValueError('call marks must be nonnegative and deltas must be in [0, 1]')
        payoff = np.maximum(self.prices[:, -1] - self.config.strike, 0.0)
        if not np.allclose(self.option_values[:, -1], payoff):
            raise ValueError('terminal option marks must equal the call payoff')


def option_marks(prices, config):
    """Return (call_values, deltas), each with shape (paths, steps + 1)."""
    values = np.empty_like(prices, dtype=float)
    deltas = np.empty_like(prices, dtype=float)
    tau = config.maturity * (1 - np.arange(config.steps) / config.steps)
    spot = prices[:, :-1]
    sigma = config.mark_vol
    d1 = (np.log(spot / config.strike)
          + (config.rate + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    values[:, :-1] = (spot * ndtr(d1)
                      - config.strike * np.exp(-config.rate * tau) * ndtr(d2))
    deltas[:, :-1] = ndtr(d1)
    values[:, -1] = np.maximum(prices[:, -1] - config.strike, 0.0)
    deltas[:, -1] = (prices[:, -1] > config.strike).astype(float)
    return values, deltas


def make_scenarios(config, paths=2000, seed=0, realized_vol=0.20,
                   gap_step=None, gap_multiplier=1.0):
    """Build fixed paths with optional one-time price gap; no future data reach the policy."""
    if type(paths) is not int or paths < 1:
        raise ValueError('paths must be a positive integer')
    if not np.isfinite(realized_vol) or realized_vol < 0:
        raise ValueError('realized_vol must be finite and nonnegative')
    if not np.isfinite(gap_multiplier) or gap_multiplier <= 0:
        raise ValueError('gap_multiplier must be finite and positive')
    if gap_step is not None:
        if type(gap_step) is not int or not 1 <= gap_step <= config.steps:
            raise ValueError('gap_step must be an integer from 1 through steps')
    elif gap_multiplier != 1.0:
        raise ValueError('a nontrivial gap_multiplier requires gap_step')
    rng = np.random.default_rng(seed)
    dt = config.maturity / config.steps
    shocks = rng.standard_normal((paths, config.steps))
    log_returns = ((config.rate - 0.5 * realized_vol**2) * dt
                   + realized_vol * np.sqrt(dt) * shocks)
    log_prices = np.concatenate([np.zeros((paths, 1)), np.cumsum(log_returns, axis=1)], axis=1)
    prices = config.spot * np.exp(log_prices)
    if gap_step is not None:
        prices[:, gap_step:] *= gap_multiplier
    values, deltas = option_marks(prices, config)
    return Scenarios(config, prices, values, deltas)
