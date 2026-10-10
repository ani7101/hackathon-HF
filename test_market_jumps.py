import unittest
import numpy as np
from market import Config
from market_jumps import JumpParameters, make_jump_scenarios, build_jump_sets
from spy_calibration import fit_spy


class JumpTests(unittest.TestCase):
    def test_reproducibility_and_event_frequency(self):
        config=Config(steps=12)
        params=JumpParameters()
        a,meta=make_jump_scenarios(config,10000,10,params)
        b,_=make_jump_scenarios(config,10000,10,params)
        np.testing.assert_array_equal(a.prices,b.prices)
        self.assertAlmostEqual(meta['mean_jumps_per_path'],2,delta=.06)
        self.assertGreater(meta['paths_with_jumps'],8000)
        self.assertLess(meta['paths_with_jumps'],9000)

    def test_no_jump_no_diffusion_expected_growth(self):
        config=Config(steps=12)
        params=JumpParameters(expected_return=.06,diffusion_vol=0,
                              jump_intensity=0,jump_log_mean=0,jump_log_std=0)
        s,_=make_jump_scenarios(config,3,10,params)
        np.testing.assert_allclose(s.prices[:,-1],100*np.exp(.06))

    def test_compensated_expected_return(self):
        s,_=make_jump_scenarios(Config(steps=12),50000,123,JumpParameters())
        self.assertAlmostEqual(s.prices[:,-1].mean(),100*np.exp(.06),delta=.8)

    def test_stress_families_and_validation(self):
        sets,meta=build_jump_sets(Config(),10,10,'stress',JumpParameters())
        self.assertEqual(set(sets),{'base','high_vol','downside_jumps','upside_jumps'})
        self.assertEqual(meta['downside_jumps']['parameters']['jump_intensity'],2)
        self.assertAlmostEqual(meta['downside_jumps']['parameters']['jump_log_mean'],-.05)
        self.assertFalse(meta['base']['induced_stress'])
        self.assertIn('synthetic',meta['base']['event_source'])
        _, severe=build_jump_sets(Config(),10,10,'stress',JumpParameters(),'severe')
        self.assertEqual(severe['downside_jumps']['parameters']['jump_intensity'],4)
        with self.assertRaises(ValueError): JumpParameters(jump_intensity=-1)

    def test_historical_proxy_fit(self):
        rng=np.random.default_rng(10)
        returns=rng.normal(.0002,.01,1000)
        returns[::100]-=.1
        prices=100*np.exp(np.r_[0,np.cumsum(returns)])
        params,meta=fit_spy(list(range(1001)),prices,prices[-1])
        self.assertGreater(params.jump_intensity,0)
        self.assertLess(params.jump_log_mean,0)
        self.assertEqual(meta['status'],'historical_threshold_fit')


if __name__=='__main__': unittest.main()
