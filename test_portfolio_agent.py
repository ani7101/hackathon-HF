"""Regression checks for bounded feedback and actionable server failures."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import portfolio_agent


class AgentTests(unittest.TestCase):
    def exercise(self, fail=False):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            if fail and len(requests) == 3:
                return httpx.Response(400, json={'error': {'message': 'context too long'}})
            policy = dict(base_band=len(requests) / 100, drawdown_response=2)
            return httpx.Response(200, json={'choices': [{'message': {
                'content': json.dumps(policy)}}]})

        real_client = httpx.Client
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(base='http://localhost:8000', model='mock',
                iterations=18, paths=8, test_paths=8, steps=8, seed=10,
                scenario_set='stress', alpha=.95, drawdown_limit=1,
                loss_limit=1000, cash_floor=-1000, output=str(Path(tmp)/'result.json'))
            with patch.object(portfolio_agent.httpx, 'Client', side_effect=lambda **kw:
                    real_client(transport=httpx.MockTransport(respond), **kw)):
                if fail:
                    with self.assertRaisesRegex(ValueError, 'context too long'):
                        portfolio_agent.run(args)
                else:
                    portfolio_agent.run(args)
            result = json.loads(Path(args.output).read_text())
        return requests, result

    def test_context_stays_bounded_and_holdout_runs(self):
        requests, result = self.exercise()
        self.assertEqual(len(requests), 18)
        self.assertTrue(all(len(r['messages']) == 2 for r in requests))
        context = json.loads(requests[-1]['messages'][1]['content'].split(
            'Simulator feedback: ', 1)[1].split('\nPropose', 1)[0])
        self.assertEqual(set(context), {'iteration', 'latest_candidate', 'candidate_memory'})
        self.assertLessEqual(len(context['candidate_memory']), 8)
        self.assertEqual(context['latest_candidate']['input_configuration']['base_band'], .17)
        self.assertEqual(context['latest_candidate']['status'], 'PASSED')
        self.assertEqual(result['simulator_evaluations'], 18)
        self.assertIsNotNone(result['final_test'])

    def test_failed_feedback_contains_only_inputs_and_compact_outcome(self):
        from portfolio import Policy
        record = dict(policy=Policy(.1, 2), feasible=False, objective=1.4,
                      normalized_violation=1.8, metrics={'base': {
                          'violations': ['cash_floor'], 'minimum_cash': -3.2}})
        feedback = portfolio_agent.feedback_summary(record)
        self.assertEqual(feedback['status'], 'FAILED')
        self.assertEqual(feedback['input_configuration'], {'base_band': .1, 'drawdown_response': 2})
        self.assertIn('cash_floor', feedback['message'])
        self.assertNotIn('metrics', feedback)

    def test_server_error_preserves_progress(self):
        _, result = self.exercise(fail=True)
        self.assertEqual(result['status'], 'interrupted')
        self.assertEqual(result['simulator_evaluations'], 2)

    def test_memory_cap_and_best_candidate_survives_long_run(self):
        from portfolio import Policy
        records = [dict(policy=Policy(i/2000, 2), feasible=False,
                        objective=1.0, normalized_violation=float(i),
                        metrics={'base': {'violations': ['cash_floor']}})
                   for i in range(1000)]
        memory = portfolio_agent.candidate_memory(records)
        self.assertLessEqual(len(memory), 8)
        self.assertLessEqual(len(json.dumps(memory, separators=(',', ':'))), 2000)
        self.assertEqual(memory[0]['b'], 0)
        self.assertTrue(any(row['b'] == .4995 for row in memory))
        self.assertEqual(memory[0]['fail'], {'base': 'C'})


if __name__ == '__main__':
    unittest.main()
