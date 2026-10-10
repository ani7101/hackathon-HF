"""Regression checks for portfolio accounting, scenario generation and risk search."""
import unittest
import numpy as np
from market import Config, Scenarios, make_scenarios, option_marks
from portfolio import Policy, simulate
from risk import Limits, cvar, evaluate
from search import optimize


class PortfolioTests(unittest.TestCase):
    def test_seed_and_gap(self):
        cfg = Config(steps=4)
        base = make_scenarios(cfg, paths=10, seed=12)
        repeated = make_scenarios(cfg, paths=10, seed=12)
        stressed = make_scenarios(cfg, paths=10, seed=12, gap_step=2, gap_multiplier=.75)
        np.testing.assert_array_equal(base.prices, repeated.prices)
        np.testing.assert_array_equal(base.prices[:, :2], stressed.prices[:, :2])
        np.testing.assert_allclose(base.prices[:, 2:] * .75, stressed.prices[:, 2:])
        self.assertFalse(np.array_equal(base.prices, make_scenarios(cfg, 10, 13).prices))
        np.testing.assert_allclose(base.option_values[:, -1], np.maximum(base.prices[:, -1]-cfg.strike, 0))

    def test_hand_calculated_cash_settlement_and_fees(self):
        cfg = Config(steps=1, rate=0, fee_rate=.01)
        paths = Scenarios(cfg, np.array([[100., 120.]]), np.array([[10., 20.]]), np.array([[.5, 1.]]))
        result = simulate(Policy(.08), paths)
        # Cash: 110 - 50 - .50 + 60 - .60 - 20 = 98.90.
        self.assertAlmostEqual(result['fees'][0], 1.10)
        self.assertAlmostEqual(result['equity'][0, -1], 98.90)
        self.assertAlmostEqual(result['cash'][0, -1], 98.90)
        self.assertAlmostEqual(result['equity'][0, 0], cfg.capital)
        self.assertAlmostEqual(result['equity'][0, 2], 99.5)
        self.assertEqual(result['equity'].shape, result['cash'].shape)
        self.assertEqual(result['equity'].shape[1], len(result['times']))

    def test_no_future_information_changes_earlier_trades(self):
        cfg = Config(steps=3)
        a = make_scenarios(cfg, 12, 3)
        prices = a.prices.copy()
        prices[:, -1] *= 2
        marks, deltas = option_marks(prices, cfg)
        b = Scenarios(cfg, prices, marks, deltas)
        x, y = simulate(Policy(.06, 2), a), simulate(Policy(.06, 2), b)
        np.testing.assert_array_equal(x['equity'][:, :-2], y['equity'][:, :-2])
        np.testing.assert_array_equal(x['cash'][:, :-2], y['cash'][:, :-2])

    def test_cvar_fractional_tail_and_ties(self):
        self.assertAlmostEqual(cvar([0, 10, 20, 30], .625), 80/3)
        self.assertAlmostEqual(cvar([0, 10, 20, 30], .75), 30)
        self.assertAlmostEqual(cvar([7, 7, 7], .95), 7)

    def test_interim_drawdown_is_not_hidden_by_recovery(self):
        def trajectory(equity):
            return dict(equity=np.array([equity], float), cash=np.full((1, 3), 100.),
                        fees=np.array([1.]), terminal_target=100.)
        limits = Limits(max_drawdown_cvar=.1)
        deep = evaluate(trajectory([100, 70, 105]), limits)
        shallow = evaluate(trajectory([100, 97, 104]), limits)
        self.assertAlmostEqual(deep['drawdown_cvar'], .3)
        self.assertEqual(deep['terminal_loss_cvar'], 0)
        self.assertFalse(deep['feasible'])
        self.assertTrue(shallow['feasible'])

    def test_cash_breach_and_insolvency_reject_policy(self):
        result = dict(equity=np.array([[100., -1., 110.]]), cash=np.array([[100., -2., 110.]]),
                      fees=np.array([1.]), terminal_target=100.)
        grade = evaluate(result, Limits(max_drawdown_cvar=2, max_terminal_loss_cvar=100))
        self.assertEqual(grade['violations'], ['cash_floor', 'insolvency'])

    def test_selects_minimum_feasible_and_reports_infeasibility(self):
        scenarios = {'base': make_scenarios(Config(steps=4), 20, 3)}
        policies = [Policy(.01), Policy(.15), Policy(.1, 2)]
        relaxed = Limits(max_drawdown_cvar=2, max_terminal_loss_cvar=1000, min_cash=-1000)
        best, records = optimize(policies, scenarios, relaxed)
        self.assertIsNotNone(best)
        self.assertAlmostEqual(best['objective'], min(x['objective'] for x in records if x['feasible']))
        best, records = optimize(policies, scenarios, Limits(min_cash=1000))
        self.assertIsNone(best)
        self.assertEqual(len(records), len(policies))
        self.assertTrue(all(not row['feasible'] for row in records))

    def test_invalid_inputs(self):
        for kwargs in ({'capital': 0}, {'steps': 0}, {'mark_vol': float('nan')}):
            with self.assertRaises(ValueError): Config(**kwargs)
        with self.assertRaises(ValueError): Policy(-.1)
        with self.assertRaises(ValueError): Policy(.1, -1)
        with self.assertRaises(ValueError): Limits(alpha=1)
        with self.assertRaises(ValueError): cvar([], .95)
        with self.assertRaises(ValueError): cvar([float('nan')], .95)
        with self.assertRaises(ValueError): make_scenarios(Config(), paths=0)
        with self.assertRaises(ValueError): make_scenarios(Config(), gap_multiplier=.5)
        with self.assertRaises(ValueError): optimize([Policy(.1)], {}, Limits())


if __name__ == '__main__':
    unittest.main()
