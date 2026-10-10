#!/usr/bin/env python3
"""AI hedge search with configurable illustrative equity values and random jumps."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import httpx

from market import Config
from portfolio import Policy
from risk import Limits
from search import optimize
from market_jumps import JumpParameters, build_jump_sets
from spy_calibration import load_spy


def parse_policy(reply):
    text = reply.strip()
    if text.startswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {'base_band', 'drawdown_response'}:
        raise ValueError('Return exactly base_band and drawdown_response.')
    if any(type(v) not in (int, float) for v in data.values()):
        raise ValueError('Both parameters must be numbers.')
    policy = Policy(**data)
    if policy.drawdown_response > 20:
        raise ValueError('drawdown_response must be in [0, 20].')
    return policy


def violation(record, limits, capital):
    return sum(
        max(0, m['drawdown_cvar'] - limits.max_drawdown_cvar)
        / max(limits.max_drawdown_cvar, 1e-6)
        + max(0, m['terminal_loss_cvar'] - limits.max_terminal_loss_cvar)
        / max(limits.max_terminal_loss_cvar, 1e-6)
        + max(0, limits.min_cash - m['minimum_cash']) / capital
        + (1 + abs(m['minimum_equity']) / capital if m['minimum_equity'] <= 0 else 0)
        for m in record['metrics'].values())


def feedback_summary(record):
    """One candidate's inputs and compact outcome; never include its history."""
    failures = {name: metrics['violations'] for name, metrics in record['metrics'].items()
                if metrics['violations']}
    return dict(input_configuration=asdict(record['policy']),
                status='PASSED' if record['feasible'] else 'FAILED',
                message=('All scenario constraints passed.' if record['feasible']
                         else 'Constraint violations: ' + json.dumps(failures)),
                mean_fees=round(record['objective'], 6),
                normalized_violation=round(record['normalized_violation'], 6))


def candidate_memory(records):
    """At most eight rows and 2,000 characters, regardless of trial count."""
    rank = lambda r: (not r['feasible'], r['objective'] if r['feasible']
                      else r['normalized_violation'])
    chosen = sorted(records, key=rank)[:3] + list(reversed(records[-5:]))
    rows, seen = [], set()
    codes = {'drawdown_cvar': 'D', 'terminal_loss_cvar': 'L',
             'cash_floor': 'C', 'insolvency': 'I'}
    for record in chosen:
        policy = record['policy']
        key = (policy.base_band, policy.drawdown_response)
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(b=round(policy.base_band, 6),
                         k=round(policy.drawdown_response, 6),
                         status='P' if record['feasible'] else 'F',
                         fee=round(record['objective'], 4),
                         v=round(record['normalized_violation'], 4),
                         fail={name: ''.join(codes[v] for v in m['violations'])
                               for name, m in record['metrics'].items() if m['violations']}))
    while len(json.dumps(rows, separators=(',', ':'))) > 2000:
        rows.pop()
    return rows


def run(args):
    calibration = {"symbol": "SPY", "status": "illustrative_not_calibrated"}
    calibrated_params = None
    if args.calibrate_spy or args.spy_csv:
        calibrated_params, calibration = load_spy(args.spy_csv, args.jump_threshold)
        args.spot = calibration["last_close"]
        args.strike = args.spot
        args.mark_vol = calibrated_params.diffusion_vol
        print("SPY calibration:", json.dumps(calibration), flush=True)
    config = Config(steps=args.steps, capital=args.capital, spot=args.spot,
                    strike=args.strike, maturity=args.maturity, rate=args.rate,
                    mark_vol=args.mark_vol, fee_rate=args.fee_rate)
    jump_params = JumpParameters(args.expected_return, args.diffusion_vol,
                                 args.jump_intensity, args.jump_log_mean, args.jump_log_std)
    limits = Limits(args.alpha, args.drawdown_limit, args.loss_limit, args.cash_floor)
    if calibrated_params is not None:
        jump_params = calibrated_params
    sets, diagnostics = build_jump_sets(config, args.paths, args.seed, args.scenario_set, jump_params, args.stress_profile)
    base = args.base.rstrip('/')
    if not base.endswith('/v1'):
        base += '/v1'
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    messages = [{'role': 'system', 'content': (
        'Optimize a short-call portfolio hedge using measured simulator feedback. '
        'Return ONLY JSON with numeric base_band in [0,0.5] and '
        'drawdown_response in [0,20]. Band = base_band/(1+drawdown_response*drawdown). '
        'Trading rebalances to fixed configured-volatility Black-Scholes delta when the share gap exceeds '
        'the band. No other policy changes are allowed. First satisfy all constraints '
        'in every scenario; then minimize base mean fees. For infeasible proposals, '
        'reduce normalized_violation. Do not repeat previous parameters. '
        'Compressed memory lists up to three best and five recent candidates, '
        'deduplicated: b=base_band, k=drawdown_response, status P=PASSED/F=FAILED, '
        'fee=base mean fees, v=normalized violation (lower is better), '
        'fail codes D=drawdown, L=terminal loss, C=cash, I=insolvency. '
        'Physical-measure expected returns, Poisson jump times, normal log jump sizes; '
        'no fixed shock day. Check calibration metadata; risk limits and fees are assumptions. '
        'Cash breaches are measured after trades, not prevented. '
        + json.dumps({'config': asdict(config), 'limits': asdict(limits), 'calibration': calibration, 'jump_families': {name: d['parameters'] for name,d in diagnostics.items()}}))}]
    records, attempts, seen = [], [], set()
    best = None
    latest_feedback = None
    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    def save(status, final_test=None):
        acceptable = [r for r in records if r['feasible']]
        selected = min(acceptable, key=lambda r: r['objective']) if acceptable else None
        closest = min(records, key=lambda r: r['normalized_violation']) if records else None
        payload = dict(status=status, search_method='ai_jump_diffusion', calibration_status=calibration['status'], calibration=calibration,
                       jump_model='compound_poisson_lognormal_physical_measure',
                       market_diagnostics=diagnostics, stress_profile=args.stress_profile, model=args.model,
                       config=asdict(config), limits=asdict(limits), seed=args.seed,
                       paths_per_set=args.paths, scenarios=list(sets),
                       objective='mean execution fees on base scenarios',
                       proposal_budget=args.iterations, simulator_evaluations=len(records),
                       best=selected, closest_infeasible=closest if selected is None else None,
                       records=records, attempts=attempts, token_usage=usage,
                       final_test=final_test,
                       assumptions=['Physical-measure jump diffusion; see calibration metadata and caveats.',
                                    'Fixed Black-Scholes marks/delta; no jump-aware option pricing.',
                                    'Single share short call; symmetric cash interest; proportional fees.',
                                    'Independent jumps; no volatility clustering, dividends or margin model.',
                                    'Finite empirical evaluation; no convergence guarantee.'])
        output.write_text(json.dumps(payload, indent=2, default=asdict, allow_nan=False) + '\n')
        return selected

    with httpx.Client(timeout=300) as client:
        if not args.model:
            response = client.get(base + '/models')
            response.raise_for_status()
            args.model = response.json()['data'][0]['id']
        for iteration in range(1, args.iterations + 1):
            # Bounded summaries retain useful search memory without growing transcripts.
            context = dict(iteration=iteration, latest_candidate=latest_feedback,
                           candidate_memory=candidate_memory(records))
            messages = [messages[0], {'role': 'user', 'content':
                        'Simulator feedback: ' + json.dumps(context, separators=(',', ':'))
                        + '\nPropose the next policy.'}]
            response = client.post(base + '/chat/completions', json=dict(
                model=args.model, messages=messages, temperature=0.6, max_tokens=512,
                chat_template_kwargs={'enable_thinking': False}))
            if response.is_error:
                save('interrupted')
                raise ValueError(f'Model server HTTP {response.status_code}: '
                                 f'{response.text[:2000]}. '
                                 f'Completed results preserved in {output.resolve()}')
            body = response.json()
            for key in usage:
                usage[key] += body.get('usage', {}).get(key, 0) or 0
            reply = body['choices'][0]['message'].get('content') or ''
            attempt = {'iteration': iteration, 'reply': reply}
            proposed_configuration = None
            try:
                policy = parse_policy(reply)
                proposed_configuration = asdict(policy)
                key = (policy.base_band, policy.drawdown_response)
                if key in seen:
                    raise ValueError('Duplicate policy; propose a different policy.')
                seen.add(key)
                _, evaluated = optimize([policy], sets, limits)
                record = evaluated[0]
                record['iteration'] = iteration
                record['normalized_violation'] = violation(record, limits, config.capital)
                records.append(record)
                latest_feedback = feedback_summary(record)
                attempt['evaluated'] = True
                print(f'Iteration {iteration}: {policy}; feasible={record["feasible"]}; '
                      f'fees={record["objective"]:.4f}; violation={record["normalized_violation"]:.4f}', flush=True)
            except (ValueError, TypeError) as exc:
                latest_feedback = dict(input_configuration=proposed_configuration,
                                       status='FAILED', message=str(exc))
                attempt.update(evaluated=False, error=str(exc))
                print(f'Iteration {iteration}: rejected proposal: {exc}', flush=True)
            attempts.append(attempt)
            best = save('running')
    final_test = None
    if best is not None:
        fresh, test_diagnostics = build_jump_sets(config, args.test_paths, args.seed + 1_000_000,
                                                      args.scenario_set, jump_params, args.stress_profile)
        _, checked = optimize([best['policy']], fresh, limits)
        final_test = dict(seed=args.seed + 1_000_000, paths_per_set=args.test_paths,
                          feasible=checked[0]['feasible'], metrics=checked[0]['metrics'],
                          market_diagnostics=test_diagnostics)
    save('selected' if best else 'no_feasible_policy', final_test)
    print(f'Results: {output.resolve()}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://localhost:8000/v1')
    parser.add_argument('--model', help='Defaults to the first model served by /v1/models.')
    parser.add_argument('--iterations', type=int, default=15)
    parser.add_argument('--paths', type=int, default=2000)
    parser.add_argument('--test-paths', type=int, default=2000)
    parser.add_argument('--steps', type=int, default=252)
    parser.add_argument('--capital', type=float, default=1000,
                        help='Initial investor capital per one-share short call.')
    parser.add_argument('--calibrate-spy', action='store_true', help='Fetch five years of SPY daily returns; fit threshold jump proxy.')
    parser.add_argument('--spy-csv', help='Offline CSV with Date, Close, Adj Close columns.')
    parser.add_argument('--jump-threshold', type=float, default=3)
    parser.add_argument('--spot', type=float, default=100)
    parser.add_argument('--strike', type=float, default=100)
    parser.add_argument('--maturity', type=float, default=1)
    parser.add_argument('--rate', type=float, default=.04)
    parser.add_argument('--mark-vol', type=float, default=.20)
    parser.add_argument('--fee-rate', type=float, default=.0005)
    parser.add_argument('--expected-return', type=float, default=.06)
    parser.add_argument('--diffusion-vol', type=float, default=.20)
    parser.add_argument('--jump-intensity', type=float, default=2)
    parser.add_argument('--jump-log-mean', type=float, default=-.04)
    parser.add_argument('--jump-log-std', type=float, default=.08)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--stress-profile', choices=('simple', 'mild', 'severe'), default='simple',
                        help='Simple adds one stress: 25% more jumps, unchanged jump sizes. Mild/severe retain prior scenarios.')
    parser.add_argument('--scenario-set', choices=('base', 'stress'), default='stress')
    parser.add_argument('--alpha', type=float, default=.95)
    parser.add_argument('--drawdown-limit', type=float, default=.10)
    parser.add_argument('--loss-limit', type=float, default=100)
    parser.add_argument('--cash-floor', type=float, default=0)
    parser.add_argument('--output', default='artifacts/portfolio-ai-jumps.json')
    args = parser.parse_args()
    if min(args.iterations, args.paths, args.test_paths, args.steps) < 1 or args.seed < 0:
        parser.error('Counts must be positive; seed must be nonnegative.')
    try:
        run(args)
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
        parser.exit(1, f'AI run stopped: {exc}\n')


if __name__ == '__main__':
    main()
