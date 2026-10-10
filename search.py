#!/usr/bin/env python3
"""Finite policy search with drawdown, terminal-loss and cash constraints.

python search.py --paths 2000 --output artifacts/portfolio-search.json
Stress sets impose separate constraints, not an assumed mixture probability.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from market import Config, make_scenarios
from portfolio import Policy, simulate
from risk import Limits, evaluate


def optimize(policies, scenario_sets, limits):
    """Return (best_record_or_None, all_records); equal-cost ties retain the first."""
    if 'base' not in scenario_sets:
        raise ValueError('scenario_sets must include base for the cost objective')
    if any(s.config != scenario_sets['base'].config for s in scenario_sets.values()):
        raise ValueError('all scenario sets must use the same portfolio configuration')
    records = []
    for policy in policies:
        metrics = {name: evaluate(simulate(policy, scenarios), limits)
                   for name, scenarios in scenario_sets.items()}
        records.append(dict(policy=policy, objective=metrics['base']['mean_fees'],
                            feasible=all(m['feasible'] for m in metrics.values()), metrics=metrics))
    if not records:
        raise ValueError('at least one candidate policy is required')
    acceptable = [record for record in records if record['feasible']]
    best = min(acceptable, key=lambda record: record['objective']) if acceptable else None
    return best, records


def build_scenario_sets(config, paths, seed, scenario_set='stress'):
    """Build fixed paths once, then reuse them for every candidate."""
    if scenario_set not in ('base', 'stress'):
        raise ValueError('scenario_set must be base or stress')
    scenarios = {'base': make_scenarios(config, paths, seed, realized_vol=0.20)}
    if scenario_set == 'stress':
        step = max(1, config.steps // 4)
        scenarios.update({
            'high_vol': make_scenarios(config, paths, seed + 1, realized_vol=0.35),
            'gap_down': make_scenarios(config, paths, seed + 2, gap_step=step, gap_multiplier=0.75),
            'gap_up': make_scenarios(config, paths, seed + 3, gap_step=step, gap_multiplier=1.25),
        })
    return scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=int, default=2000, help='search paths per scenario set')
    parser.add_argument('--test-paths', type=int, default=2000, help='fresh paths for the selected policy')
    parser.add_argument('--steps', type=int, default=252)
    parser.add_argument('--capital', type=float, default=100)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--scenario-set', choices=('base', 'stress'), default='stress')
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--drawdown-limit', type=float, default=0.10, help='fraction of peak equity')
    parser.add_argument('--loss-limit', type=float, default=10.0, help='money units')
    parser.add_argument('--cash-floor', type=float, default=0.0)
    parser.add_argument('--output', default='artifacts/portfolio-search.json')
    args = parser.parse_args()
    if min(args.paths, args.test_paths, args.steps) < 1 or args.seed < 0:
        parser.error('path counts and steps must be positive; seed must be nonnegative')
    try:
        config = Config(steps=args.steps, capital=args.capital)
        limits = Limits(args.alpha, args.drawdown_limit, args.loss_limit, args.cash_floor)
    except ValueError as exc:
        parser.error(str(exc))
    policies = [Policy(b, k) for b in (0.0, 0.03, 0.06, 0.10, 0.15) for k in (0.0, 2.0, 5.0)]
    print(f'Searching {len(policies)} policies; {args.paths} paths per {args.scenario_set} scenario set.', flush=True)
    scenarios = build_scenario_sets(config, args.paths, args.seed, args.scenario_set)
    best, records = optimize(policies, scenarios, limits)
    print(f'Feasible candidates: {sum(record["feasible"] for record in records)}/{len(records)}')
    final_test = None
    if best is None:
        print('No feasible policy in this candidate set. See violations in the output JSON.')
    else:
        print(f'Chosen policy: {best["policy"]}; base mean fees: {best["objective"]:.6f}')
        test_seed = args.seed + 1_000_000
        fresh = build_scenario_sets(config, args.test_paths, test_seed, args.scenario_set)
        metrics = {name: evaluate(simulate(best['policy'], paths), limits) for name, paths in fresh.items()}
        final_test = dict(seed=test_seed, paths_per_set=args.test_paths, metrics=metrics,
                          feasible=all(m['feasible'] for m in metrics.values()))
        print('Fresh-scenario evaluation: ' + ('PASS' if final_test['feasible'] else 'FAIL'))
        # Report fresh results without selecting another candidate on that test set.
    payload = dict(status='selected' if best else 'no_feasible_policy', config=asdict(config),
                   limits=asdict(limits), seed=args.seed, paths_per_set=args.paths,
                   scenario_set=args.scenario_set, scenarios=list(scenarios),
                   objective='mean execution fees on base scenarios',
                   best=best, records=records, final_test=final_test,
                   assumptions=['One short call on one share; cash settlement and stock liquidation.',
                                'Fixed mark_vol; realized volatility and price gaps vary by named scenario.',
                                'Symmetric cash interest, no borrowing spread or margin/redemption model.',
                                'Stress sets have no assigned mixture probabilities.',
                                'Finite-grid empirical feasibility, not a market-wide guarantee.'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=asdict, allow_nan=False) + '\n')
    print(f'Results saved to {output.resolve()}')


if __name__ == '__main__':
    main()
