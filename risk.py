"""Continuous cost and tail-risk measurements, separate from simulation and search."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Limits:
    alpha: float = 0.95
    max_drawdown_cvar: float = 0.10
    max_terminal_loss_cvar: float = 10.0
    min_cash: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError('alpha must be finite and between 0 and 1')
        for name in ('max_drawdown_cvar', 'max_terminal_loss_cvar'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not np.isfinite(self.min_cash):
            raise ValueError('min_cash must be finite')


def cvar(losses, alpha):
    """Empirical upper-tail CVaR, correctly weighting a fractional tail observation."""
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or not losses.size or not np.isfinite(losses).all():
        raise ValueError('losses must be a nonempty finite one-dimensional array')
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')
    threshold = np.quantile(losses, alpha, method='inverted_cdf')
    return float(threshold + np.maximum(losses - threshold, 0).mean() / (1 - alpha))


def evaluate(trajectory, limits):
    """Return metrics and explicit violations; feasibility applies only to these paths."""
    equity = np.asarray(trajectory['equity'])
    cash = np.asarray(trajectory['cash'])
    fees = np.asarray(trajectory['fees'])
    if (equity.ndim != 2 or not equity.size or cash.shape != equity.shape
            or fees.shape != (equity.shape[0],)):
        raise ValueError('equity, cash and fees have incompatible shapes')
    if not all(np.isfinite(x).all() for x in (equity, cash, fees)):
        raise ValueError('trajectory contains nonfinite values')
    if np.any(equity[:, 0] <= 0) or np.any(fees < 0):
        raise ValueError('initial equity must be positive and fees nonnegative')
    target = trajectory['terminal_target']
    if not np.isfinite(target):
        raise ValueError('terminal_target must be finite')
    running_peak = np.maximum.accumulate(equity, axis=1)
    drawdowns = (running_peak - equity) / running_peak
    maximum_drawdowns = drawdowns.max(axis=1)
    terminal_losses = np.maximum(target - equity[:, -1], 0)
    drawdown_tail = cvar(maximum_drawdowns, limits.alpha)
    terminal_tail = cvar(terminal_losses, limits.alpha)
    minimum_cash, minimum_equity = float(cash.min()), float(equity.min())
    violations = []
    if drawdown_tail > limits.max_drawdown_cvar:
        violations.append('drawdown_cvar')
    if terminal_tail > limits.max_terminal_loss_cvar:
        violations.append('terminal_loss_cvar')
    if minimum_cash < limits.min_cash:
        violations.append('cash_floor')
    if minimum_equity <= 0:
        violations.append('insolvency')
    return dict(mean_fees=float(fees.mean()), drawdown_cvar=drawdown_tail,
                terminal_loss_cvar=terminal_tail, minimum_cash=minimum_cash,
                minimum_equity=minimum_equity, mean_terminal_equity=float(equity[:, -1].mean()),
                feasible=not violations, violations=violations)
