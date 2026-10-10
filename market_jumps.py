"""Physical-measure compound-Poisson/lognormal jumps; illustrative, not calibrated."""
from dataclasses import dataclass, asdict
import numpy as np
from market import Scenarios, option_marks


@dataclass(frozen=True)
class JumpParameters:
    expected_return: float = .06
    diffusion_vol: float = .20
    jump_intensity: float = 2.0
    jump_log_mean: float = -.04
    jump_log_std: float = .08

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Jump parameters must be finite.')
        if min(self.diffusion_vol, self.jump_intensity, self.jump_log_std) < 0:
            raise ValueError('Volatility, jump intensity and jump std must be nonnegative.')
        if abs(self.jump_log_mean) > 1 or self.jump_log_std > 1 or self.jump_intensity > 1000:
            raise ValueError('Jump parameters exceed supported simulation bounds.')


def make_jump_scenarios(config, paths, seed, params):
    """Aggregate any number of jumps per day exactly for normal log jump sizes."""
    if type(paths) is not int or paths < 1:
        raise ValueError('paths must be a positive integer')
    dt = config.maturity / config.steps
    rng = np.random.default_rng(seed)
    shape = (paths, config.steps)
    counts = rng.poisson(params.jump_intensity * dt, size=shape)
    jump_logs = (counts * params.jump_log_mean
                 + np.sqrt(counts) * params.jump_log_std * rng.standard_normal(shape))
    expected_jump = np.expm1(params.jump_log_mean + .5 * params.jump_log_std**2)
    drift = params.expected_return - params.jump_intensity * expected_jump
    increments = ((drift - .5 * params.diffusion_vol**2) * dt
                  + params.diffusion_vol * np.sqrt(dt) * rng.standard_normal(shape)
                  + jump_logs)
    prices = config.spot * np.exp(np.column_stack([
        np.zeros(paths), np.cumsum(increments, axis=1)]))
    values, deltas = option_marks(prices, config)
    scenarios = Scenarios(config, prices, values, deltas)
    diagnostics = dict(parameters=asdict(params), seed=seed,
                       observed_jumps=int(counts.sum()),
                       paths_with_jumps=int(np.count_nonzero(counts.sum(axis=1))),
                       mean_jumps_per_path=float(counts.sum(axis=1).mean()))
    return scenarios, diagnostics


def build_jump_sets(config, paths, seed, scenario_set, params):
    families = {'base': params}
    if scenario_set == 'stress':
        families.update(
            high_vol=JumpParameters(**{**asdict(params), 'diffusion_vol': params.diffusion_vol*1.75}),
            downside_jumps=JumpParameters(**{**asdict(params),
                'jump_intensity': params.jump_intensity*2,
                'jump_log_mean': params.jump_log_mean-.08}),
            upside_jumps=JumpParameters(**{**asdict(params),
                'jump_intensity': params.jump_intensity*2,
                'jump_log_mean': params.jump_log_mean+.12}))
    elif scenario_set != 'base':
        raise ValueError('scenario_set must be base or stress')
    sets, diagnostics = {}, {}
    for offset, (name, family) in enumerate(families.items()):
        sets[name], diagnostics[name] = make_jump_scenarios(config, paths, seed+offset, family)
    return sets, diagnostics
