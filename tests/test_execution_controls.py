import json
import unittest
from unittest.mock import patch

from agents.tool_context import ToolContext
from scripts.researchers import (
    ISSUER, ResearchTrace, _grounded_claim_supports,
    build_researcher_agents, run_researcher,
    normalize_issuer_rules,
    calculate_registered_terms,
)
from scripts.lead import ResearchRun, build_lead_agent


class EvidenceTests(unittest.TestCase):
    def registered(self, values):
        return [{'name': 'register_issuer_terms', 'result': {'evidence': [
            {'card_id': 'example', 'field': key, 'value': value,
             'status': 'issuer_grounded', 'evidence_id': key}
            for key, value in values.items()
        ]}}]

    def test_scoped_calculation_requires_only_spent_categories(self):
        calls = self.registered({'annual_fee_usd': 0, 'dining_rate_pct': 3,
                                 'dining_cap_status': 'uncapped', 'other_rate_pct': 1.5,
                                 'other_cap_status': 'uncapped'})
        result = calculate_registered_terms('example', {'dining': 500, 'other': 300}, calls)
        self.assertEqual(result['status'], 'estimated')
        self.assertEqual(result['net_cash_back_usd'], 234)
        self.assertNotIn('bonus_categories', result['required_fields'])

    def test_missing_cap_preserves_other_category_without_publishing_net(self):
        calls = self.registered({'annual_fee_usd': 0, 'dining_rate_pct': 3,
                                 'other_rate_pct': 1.5, 'other_cap_status': 'uncapped'})
        result = calculate_registered_terms('example', {'dining': 500, 'other': 300}, calls)
        self.assertEqual(result['status'], 'incomplete')
        self.assertNotIn('net_cash_back_usd', result)
        self.assertEqual(result['categories'][1]['cash_back_usd'], 54)
        scenario = calculate_registered_terms('example', {'dining': 500, 'other': 300}, calls, ['dining'])
        self.assertEqual(scenario['status'], 'conditional_estimate')
        self.assertEqual(scenario['net_cash_back_usd'], 234)

    def test_uniform_registered_rate_covers_all_nonzero_spending(self):
        calls = self.registered({'annual_fee_usd': 0, 'all_rate_pct': 2, 'all_cap_status': 'uncapped'})
        result = calculate_registered_terms('example', {'dining': 500, 'other': 300}, calls)
        self.assertEqual(result['net_cash_back_usd'], 192)

    def test_normalization_keeps_good_fields_when_another_field_fails(self):
        calls = [{'name': 'search_official_terms', 'result': {'evidence': [{
            'card_id': 'example', 'claim_id': 'fee', 'field': 'issuer_grounded_claim',
            'value': 'Annual fee: $0', 'status': 'issuer_grounded',
        }]}}]
        result = normalize_issuer_rules('example', [
            {'field': 'annual_fee_usd', 'value': 0, 'claim_id': 'fee'},
            {'field': 'other_rate_pct', 'value': 9, 'claim_id': 'fee'},
        ], calls)
        self.assertEqual(result['status'], 'semantic_mismatch')
        self.assertEqual(result['evidence'][0]['field'], 'annual_fee_usd')
        self.assertEqual(result['validation_errors'][0]['field'], 'other_rate_pct')

    def test_budget_state_reports_individual_tool_exhaustion(self):
        trace = ResearchTrace(tool_counts={'search_official_terms': 8})
        self.assertEqual(trace.budgets()['tools']['search_official_terms'], 0)
        self.assertEqual(trace.budgets()['tools']['calculate_verified_rewards'], 8)

    def test_absence_of_cap_is_not_proof_of_no_cap(self):
        self.assertFalse(_grounded_claim_supports(
            'capped_categories', [], 'Terms do not specify a spending cap.'))
        self.assertTrue(_grounded_claim_supports(
            'capped_categories', [], 'Unlimited 2% cash rewards on net purchases.'))

    def test_rate_cannot_be_borrowed_from_another_category(self):
        claim = 'Dining earns 3%. Other purchases earn 1.5%.'
        self.assertTrue(_grounded_claim_supports('dining_rate_pct', 3, claim))
        self.assertFalse(_grounded_claim_supports('dining_rate_pct', 1.5, claim))

    def test_uniform_net_purchase_rate_is_supported(self):
        self.assertTrue(_grounded_claim_supports(
            'other_rate_pct', 2, 'Unlimited 2% on all net purchases.'))
        self.assertTrue(_grounded_claim_supports(
            'other_rate_pct', 2,
            'Unlimited 2% cash rewards on Visa purchases (net purchases defined as purchases minus returns/credits)'))

    def test_duplicate_guard_does_not_overwrite_calculation_failure(self):
        state = ResearchRun(hooks=None, researcher_agents={})
        state.research_results = [{
            'agent': ISSUER, 'card_ids': ['example'], 'task': 'Calculate net rewards',
            'tool_calls': [
                {'name': 'calculate_verified_rewards', 'args': {'card_id': 'example'},
                 'result': {'status': 'semantic_mismatch', 'field': 'other_rate_pct'}},
                {'name': 'calculate_verified_rewards', 'args': {'card_id': 'example'},
                 'result': {'status': 'duplicate_blocked'}},
            ],
        }]
        self.assertEqual(state.research_state()['calculations_by_card']['example']['field'], 'other_rate_pct')

    def test_successful_followup_closes_old_failed_assignment(self):
        state = ResearchRun(hooks=None, researcher_agents={})
        state.research_results = [
            {'agent': ISSUER, 'card_ids': ['wf-active-cash'], 'status': 'failed',
             'task': 'Find fee', 'unresearched_card_ids': ['wf-active-cash']},
            {'agent': ISSUER, 'card_ids': ['wf-active-cash'], 'status': 'completed',
             'task': 'Find fee', 'unresearched_card_ids': []},
        ]
        self.assertEqual(state.evidence_quality_issues(), [])


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_exhausted_researcher_is_not_dispatched_again(self):
        tool = next(tool for tool in build_lead_agent('test-model').tools
                    if tool.name == 'delegate_issuer_verification')
        state = ResearchRun(hooks=None, researcher_agents={})
        state.researcher_traces[ISSUER] = ResearchTrace(tool_attempts=24)
        args = json.dumps({'card_ids': ['wf-active-cash'], 'task': 'Follow up', 'round_number': 2})
        ctx = ToolContext(context=state, tool_name=tool.name, tool_call_id='empty', tool_arguments=args)
        with patch('scripts.lead.run_researcher') as run:
            result = await tool.on_invoke_tool(ctx, args)
        run.assert_not_called()
        self.assertTrue(result['do_not_retry'])
        self.assertEqual(state.researcher_call_count, 0)

    def test_live_issuer_has_only_one_calculation_path(self):
        agent = build_researcher_agents('test-model')[ISSUER]
        names = {tool.name for tool in agent.tools}
        self.assertEqual(names, {'search_official_terms', 'lookup_issuer_terms',
                                 'register_issuer_terms', 'calculate_from_terms'})
        self.assertNotIn('use normalize_issuer_rules', agent.instructions)

    async def test_non_snapshot_cards_do_not_offer_empty_fallback(self):
        agent = build_researcher_agents('test-model')[ISSUER]
        async def done(actual_agent, prompt, **kwargs):
            self.assertNotIn('lookup_issuer_terms', {tool.name for tool in actual_agent.tools})
            self.assertEqual(json.loads(prompt)['snapshot_available_card_ids'], [])
            return type('Result', (), {'final_output': 'Done'})()
        with patch('agents.Runner.run', side_effect=done):
            await run_researcher(agent, ['wf-active-cash'], 'Find fee')

    async def test_unchanged_calculation_is_cached_and_new_terms_allow_recalculation(self):
        tool = next(tool for tool in build_researcher_agents('test-model')[ISSUER].tools
                    if tool.name == 'calculate_from_terms')
        trace = ResearchTrace(tool_calls=[{'name': 'register_issuer_terms', 'result': {'evidence': [
            {'card_id': 'example', 'field': name, 'value': value, 'status': 'issuer_grounded'}
            for name, value in {'annual_fee_usd': 0, 'all_rate_pct': 2, 'all_cap_status': 'unknown'}.items()
        ]}}])
        arguments = json.dumps({'card_id': 'example', 'spending': {
            'grocery': 0, 'dining': 500, 'travel': 0, 'gas': 0, 'online_retail': 0,
            'transit': 0, 'streaming': 0, 'other': 300}, 'assume_uncapped': []})
        ctx = ToolContext(context=trace, tool_name=tool.name, tool_call_id='calc', tool_arguments=arguments)
        first = await tool.on_invoke_tool(ctx, arguments)
        cached = await tool.on_invoke_tool(ctx, arguments)
        self.assertEqual(first['status'], 'incomplete')
        self.assertTrue(cached['cached'])
        self.assertEqual(trace.tool_counts[tool.name], 1)
        trace.tool_calls.append({'name': 'register_issuer_terms', 'result': {'evidence': [{
            'card_id': 'example', 'field': 'all_cap_status', 'value': 'uncapped', 'status': 'issuer_grounded',
        }]}})
        recalculated = await tool.on_invoke_tool(ctx, arguments)
        self.assertEqual(recalculated['net_cash_back_usd'], 192)
        self.assertEqual(trace.tool_counts[tool.name], 2)

    async def test_quoted_terms_feed_pure_calculator(self):
        agent = build_researcher_agents('test-model')[ISSUER]
        tool = next(item for item in agent.tools if item.name == 'register_issuer_terms')
        quote = 'Annual fee $0. Unlimited 2% cash rewards on all purchases.'
        trace = ResearchTrace(tool_calls=[{
            'name': 'search_official_terms', 'result': {'evidence': [{
                'card_id': 'example', 'claim_id': 'source', 'status': 'issuer_grounded',
                'field': 'issuer_grounded_claim', 'value': quote,
            }]},
        }])
        args = json.dumps({'card_id': 'example', 'terms': [
            {'field': key, 'value': value, 'claim_id': 'source', 'quote': quote}
            for key, value in {'annual_fee_usd': 0, 'all_rate_pct': 2, 'all_cap_status': 'uncapped'}.items()
        ]})
        ctx = ToolContext(context=trace, tool_name=tool.name, tool_call_id='extract', tool_arguments=args)
        result = await tool.on_invoke_tool(ctx, args)
        self.assertEqual(result['status'], 'registered')
        calculation = calculate_registered_terms('example', {'dining': 500, 'other': 300}, trace.tool_calls)
        self.assertEqual(calculation['net_cash_back_usd'], 192)

    async def test_quoted_extraction_blocks_invented_numbers_and_bounds_repairs(self):
        agent = build_researcher_agents('test-model')[ISSUER]
        tool = next(item for item in agent.tools if item.name == 'register_issuer_terms')
        trace = ResearchTrace(tool_calls=[{
            'name': 'search_official_terms', 'result': {'evidence': [{
                'card_id': 'example', 'claim_id': 'claim', 'status': 'issuer_grounded',
                'field': 'issuer_grounded_claim', 'value': 'Unlimited 2% cash rewards on all purchases.',
            }]},
        }])
        for index, value in enumerate([9, 8, 2]):
            args = json.dumps({'card_id': 'example', 'terms': [{
                'field': 'all_rate_pct', 'value': value, 'claim_id': 'claim',
                'quote': 'Unlimited 2% cash rewards on all purchases.',
            }]})
            ctx = ToolContext(context=trace, tool_name=tool.name,
                              tool_call_id=str(index), tool_arguments=args)
            result = await tool.on_invoke_tool(ctx, args)
        self.assertEqual(result['extraction_errors'][0]['error'], 'repair_exhausted')
        self.assertEqual(result['evidence'], [])

    async def test_duplicate_tool_does_not_repeat_underlying_search(self):
        agent = build_researcher_agents('test-model')[ISSUER]
        tool = next(item for item in agent.tools if item.name == 'search_official_terms')
        trace = ResearchTrace()
        args = json.dumps({'card_id': 'wf-active-cash', 'task': 'Verify fee'})
        ctx = ToolContext(context=trace, tool_name=tool.name,
                          tool_call_id='test', tool_arguments=args)
        async def search(**kwargs):
            return {'status': 'search_unavailable', 'evidence': []}
        with patch('scripts.researchers.search_issuer_terms', side_effect=search) as mocked:
            first = await tool.on_invoke_tool(ctx, args)
            second = await tool.on_invoke_tool(ctx, args)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(first['error_kind'], 'external_unavailable')
        self.assertEqual(second['status'], 'duplicate_blocked')

    async def test_failed_run_preserves_progress_and_suppresses_model_arithmetic(self):
        agent = build_researcher_agents('test-model')[ISSUER]
        trace = ResearchTrace()
        async def fail(*args, **kwargs):
            trace.tool_calls.append({
                'name': 'search_official_terms', 'args': {'card_id': 'wf-active-cash'},
                'result': {'evidence': [{'claim_id': 'retained', 'value': 'Source claim'}]},
            })
            raise RuntimeError('max turns')
        with patch('agents.Runner.run', side_effect=fail) as runner:
            result = await run_researcher(agent, ['wf-active-cash'],
                                          'Calculate net rewards', trace=trace)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(len(result['tool_calls']), 1)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(json.loads(result['summary'])['verified_calculations'], {})

    async def test_exhausted_tool_budget_blocks_new_network_work(self):
        agent = build_researcher_agents('test-model')[ISSUER]
        tool = next(item for item in agent.tools if item.name == 'search_official_terms')
        args = json.dumps({'card_id': 'wf-active-cash', 'task': 'New search'})
        ctx = ToolContext(context=ResearchTrace(tool_attempts=24), tool_name=tool.name,
                          tool_call_id='budget', tool_arguments=args)
        with patch('scripts.researchers.search_issuer_terms') as mocked:
            result = await tool.on_invoke_tool(ctx, args)
        mocked.assert_not_called()
        self.assertEqual(result['status'], 'budget_exhausted')
