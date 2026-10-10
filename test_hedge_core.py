"""Regression checks for the replacement level-5 core and policy parser."""
import unittest
import numpy as np
from hedge_core import BadExpression, bs_call, compile_expr, last_line
from hedgesim import PathSet, default_params
import level5_hedge as level


class HedgeCoreTests(unittest.TestCase):
    def test_benchmark(self):
        self.assertAlmostEqual(float(bs_call(100, 100, 1, .05, .2)), 10.450583572185565)

    def test_vector_formula(self):
        fn = compile_expr('maximum(0.01, 2*(c*S*gamma**2)**(1/3))', level.BAND_VARS)
        values = dict(c=.005, S=np.array([100., 110.]), gamma=np.array([.02, .03]))
        expected = np.maximum(.01, 2*(values['c']*values['S']*values['gamma']**2)**(1/3))
        np.testing.assert_allclose(fn(**values), expected)

    def test_reject_unsafe_or_unbounded_syntax(self):
        for expr in ["__import__('os')", "open('/tmp/test')", 'S.__class__',
                     'S[0]', '[1]*100', 'lambda: 1', 'exp(x=1)', 'sqrt(1,2)',
                     'True', '1e999', '(1,2)', 'unknown', '2'*601]:
            with self.subTest(expr=expr), self.assertRaises(BadExpression):
                compile_expr(expr, level.BAND_VARS)

    def test_nonfinite_and_bad_shape(self):
        ps = PathSet(default_params(steps=4), 20, 1)
        for expr in ['log(-1)', '1/0', 'exp(10000)', '10**(10**10)']:
            self.assertIsNone(ps.band_from(compile_expr(expr, [])))
        self.assertIsNone(ps.band_from(lambda **env: np.zeros((3, 2))))

    def test_invalid_interval_does_not_crash(self):
        for value in ['0', '64', '1.5', '1e999']:
            with self.assertRaises(level.BadPolicy):
                level.parse_policy(level.make(1), 'policy: interval='+value)

    def test_final_line_and_thinking(self):
        self.assertEqual(last_line('<think>policy: band=0.5</think>\npolicy: band=0.1',
                                   r'policy\s*:\s*(.+)'), 'band=0.1')

    def test_scoring_thresholds(self):
        from unittest.mock import patch
        reference = dict(Jstar=1., naive={'J': 2.}, unhedged={'J': 3.},
                         summary={'cost': .5, 'risk': .5})
        summary = dict(J=1.02, cost=.5, risk=.52, trades=10.)
        class FakePaths:
            P=20
            prm={'lam': 1.}
        with patch.object(level, 'test_set', return_value=FakePaths()), \
             patch.object(level, 'reference', return_value=reference), \
             patch.object(level, '_run', return_value=summary):
            self.assertEqual(level.check(level.make(2), 'policy: band=0.1')['reward'], 1.)
            self.assertEqual(level.check(level.make(3), 'band(gamma,S,tau) = 0.1')['reward'], .8)


if __name__ == '__main__':
    unittest.main()
