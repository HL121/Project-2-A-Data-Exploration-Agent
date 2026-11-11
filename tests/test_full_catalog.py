import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as Obj
from unittest.mock import AsyncMock, patch

from issuer_search import _product_tokens, parse_grounded_response
from research_data import catalog, match_cfpb, search_cards, search_cfpb_products, shortlist_cards, survey
from scripts.researchers import research_costs, research_rewards, verify_issuer_terms


class CatalogTests(unittest.TestCase):
    def test_complete_snapshots_and_unique_ids(self):
        metadata = json.loads(Path('data/research_sources.json').read_text())
        self.assertEqual(len(catalog()), metadata['catalog_count'])
        self.assertEqual(len(survey()), metadata['cfpb_count'])
        self.assertGreater(len(catalog()), 200)
        self.assertGreater(len(survey()), 600)

    def test_non_amex_discovery_and_lookup(self):
        cards = search_cards('freedom', issuer='Chase')['cards']
        self.assertIn('chase-freedom-unlimited', [card['slug'] for card in cards])
        self.assertTrue(research_rewards('wf-active-cash')['evidence'])
        self.assertEqual(research_costs('chase-freedom-unlimited')['match']['status'], 'matched')
        self.assertEqual(research_costs('wf-active-cash')['match']['status'], 'matched')

    def test_screening_filters_inactive_business_and_fees(self):
        result = shortlist_cards({'grocery': 800, 'dining': 300}, max_annual_fee_usd=0, limit=20)
        self.assertGreater(result['scored_count'], 10)
        self.assertGreater(len({item['issuer'] for item in result['candidates']}), 3)
        for item in result['candidates']:
            card = catalog()[item['card_id']]
            self.assertEqual(card['status'], 'active')
            self.assertEqual(item['annual_fee_usd'], 0)
            self.assertNotIn('business', card['name'].casefold())
            self.assertEqual(item['status'], 'requires_issuer_verification')
        self.assertNotIn('citi-custom-cash', [item['card_id'] for item in result['candidates']])

    def test_missing_typed_rewards_are_explicit(self):
        result = research_rewards('amex-blue-business-cash')
        self.assertTrue(any('missing' in gap for gap in result['gaps']))
        self.assertIsNone(next(item['value'] for item in result['evidence'] if item['field'] == 'rewards_rates'))

    def test_no_fuzzy_match_is_accepted(self):
        card = {**catalog()['chase-freedom-unlimited'], 'name': 'Chase Freedom Unknown'}
        self.assertEqual(match_cfpb(card)['status'], 'not_found')
        self.assertTrue(match_cfpb(card)['candidates'])
        card['issuer'] = 'Unknown Bank'
        self.assertFalse(match_cfpb(card)['candidates'])

    def test_duplicate_survey_rows_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'survey.csv'
            with file.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['Institution Name', 'Product Name', 'Report Date'])
                writer.writeheader()
                row = {'Institution Name': 'Jpmorgan Chase Bank, National Association',
                       'Product Name': 'Freedom Unlimited', 'Report Date': '2025-12-31'}
                writer.writerows([row, row])
            result = match_cfpb(catalog()['chase-freedom-unlimited'], file)
        self.assertEqual(result['status'], 'ambiguous')
        self.assertEqual(result['candidate_count'], 2)

    def test_survey_search_can_find_products_outside_catalog(self):
        result = search_cfpb_products('secured', limit=30)
        self.assertGreater(result['matched_count'], 5)
        self.assertTrue(all('secured' in row['Product Name'].casefold() for row in result['records']))


class GroundingTests(unittest.IsolatedAsyncioTestCase):
    async def test_product_identity_ignores_issuer_name_boilerplate(self):
        self.assertEqual(_product_tokens(catalog()['usbank-altitude-go']), {'altitude', 'go'})
        self.assertEqual(_product_tokens(catalog()['savor']), {'savor'})
        self.assertEqual(_product_tokens(catalog()['savorone-cash-rewards']), {'savorone'})

        metadata = Obj(
            grounding_chunks=[Obj(web=Obj(
                uri='https://www.usbank.com/credit-cards/altitude-go-visa-signature-credit-card.html',
                title='U.S. Bank',
            ))],
            grounding_supports=[Obj(
                segment=Obj(text='Altitude Go earns rewards on dining.'),
                grounding_chunk_indices=[0],
            )],
            web_search_queries=[], search_entry_point=None,
        )
        result = await parse_grounded_response(
            'usbank-altitude-go', 'Us Bank', Obj(candidates=[Obj(grounding_metadata=metadata)]),
        )
        self.assertEqual(result['status'], 'grounded')

    async def test_only_official_supported_claims_are_returned(self):
        chunks = [Obj(web=Obj(uri='https://creditcards.chase.com/cash-back-credit-cards/freedom/unlimited', title='Chase')),
                  Obj(web=Obj(uri='https://chase.com.example.org/card', title='Unrelated'))]
        supports = [Obj(segment=Obj(text='Chase Freedom Unlimited has a $0 annual fee.'), grounding_chunk_indices=[0]),
                    Obj(segment=Obj(text='Unsupported rewards claim.'), grounding_chunk_indices=[1])]
        metadata = Obj(grounding_chunks=chunks, grounding_supports=supports,
                       web_search_queries=['Chase Freedom official terms'],
                       search_entry_point=Obj(rendered_content='<div>Search suggestions</div>'))
        response = Obj(candidates=[Obj(grounding_metadata=metadata)])
        result = await parse_grounded_response('chase-freedom-unlimited', 'Chase', response)
        self.assertEqual(len(result['evidence']), 1)
        self.assertEqual(result['evidence'][0]['value'], 'Chase Freedom Unlimited has a $0 annual fee.')
        self.assertEqual(result['status'], 'grounded')
        self.assertTrue(result['product_identity_verified'])
        self.assertTrue(result['search_entry_point_html'])

    async def test_ungrounded_answer_cannot_become_verified_evidence(self):
        result = await parse_grounded_response('chase-freedom-unlimited', 'Chase',
                                               Obj(candidates=[Obj(grounding_metadata=None)]))
        self.assertFalse(result['evidence'])
        self.assertEqual(result['status'], 'insufficient_grounding')

    async def test_redirected_sources_keep_the_resolved_official_url(self):
        metadata = Obj(grounding_chunks=[Obj(web=Obj(uri='https://vertexaisearch.cloud.google.com/redirect', title='Chase'))],
                       grounding_supports=[Obj(segment=Obj(text='Chase Freedom Unlimited has no annual fee.'), grounding_chunk_indices=[0])],
                       web_search_queries=[], search_entry_point=None)
        resolved = 'https://creditcards.chase.com/cash-back-credit-cards/freedom/unlimited'
        with patch('issuer_search._source_url', new=AsyncMock(return_value=resolved)):
            result = await parse_grounded_response('chase-freedom-unlimited', 'Chase',
                                                   Obj(candidates=[Obj(grounding_metadata=metadata)]))
        self.assertEqual(result['evidence'][0]['source_url'], resolved)

    async def test_comparison_page_claim_for_other_product_is_rejected(self):
        url = 'https://www.americanexpress.com/blue-cash-preferred-vs-blue-cash-everyday/'
        metadata = Obj(
            grounding_chunks=[Obj(web=Obj(uri=url, title='American Express comparison'))],
            grounding_supports=[Obj(
                segment=Obj(text='Earn 6% cash back at U.S. supermarkets.'),
                grounding_chunk_indices=[0],
            )],
            web_search_queries=[], search_entry_point=None,
        )
        result = await parse_grounded_response(
            'blue-cash-everyday', 'American Express',
            Obj(candidates=[Obj(grounding_metadata=metadata)]),
        )
        self.assertFalse(result['evidence'])
        self.assertFalse(result['product_identity_verified'])
        self.assertEqual(result['status'], 'product_identity_unverified')

    async def test_comparison_page_explicit_target_claim_is_accepted(self):
        url = 'https://www.americanexpress.com/blue-cash-preferred-vs-blue-cash-everyday/'
        metadata = Obj(
            grounding_chunks=[Obj(web=Obj(uri=url, title='American Express comparison'))],
            grounding_supports=[Obj(
                segment=Obj(text='Blue Cash Everyday earns 3% cash back at U.S. supermarkets.'),
                grounding_chunk_indices=[0],
            )],
            web_search_queries=[], search_entry_point=None,
        )
        result = await parse_grounded_response(
            'blue-cash-everyday', 'American Express',
            Obj(candidates=[Obj(grounding_metadata=metadata)]),
        )
        self.assertEqual(len(result['evidence']), 1)
        self.assertTrue(result['product_identity_verified'])


try:
    from agents.tool_context import ToolContext
except ImportError:
    ToolContext = None


@unittest.skipIf(ToolContext is None, 'Install project dependencies to exercise Agents SDK tools.')
class AgentToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_calculator_requires_headers_and_preserves_claim_provenance(self):
        from scripts.researchers import ISSUER, ResearchTrace, build_researcher_agents
        agent = build_researcher_agents('test-model', legacy_tools=True)[ISSUER]
        tool = next(tool for tool in agent.tools if tool.name == 'calculate_verified_rewards')
        self.assertTrue({'annual_fee', 'bonus_categories', 'capped_categories'}.issubset(tool.params_json_schema['required']))
        claim = {'card_id': 'wf-active-cash', 'claim_id': 'claim-1', 'field': 'issuer_grounded_claim',
                 'value': 'Unlimited 2% cash rewards on eligible purchases with no annual fee.',
                 'source_url': 'https://www.wellsfargo.com/card', 'source_date': '2026-11-10',
                 'status': 'issuer_grounded'}
        trace = ResearchTrace(tool_calls=[{'name': 'search_official_terms', 'args': {'card_id': 'wf-active-cash'},
                                          'result': {'evidence': [claim]}}])
        payload = {'card_id': 'wf-active-cash',
                   'spending': dict(grocery=0, dining=300, travel=0, gas=0, online_retail=0, transit=0, streaming=0, other=500),
                   'annual_fee': {'value': 0, 'claim_id': 'claim-1'},
                   'bonus_categories': {'value': [], 'claim_id': 'claim-1'},
                   'capped_categories': {'value': [], 'claim_id': 'claim-1'},
                   'rules': [{'field': 'other_rate_pct', 'value': 2, 'claim_id': 'claim-1'}]}
        await tool.on_invoke_tool(ToolContext(context=trace, tool_name=tool.name, tool_call_id='test-call', tool_arguments=json.dumps(payload)), json.dumps(payload))
        result = trace.tool_calls[-1]['result']
        self.assertEqual(result['net_cash_back_usd'], 192)
        self.assertTrue(all(item['claim_id'] == 'claim-1' for item in result['evidence']))
        payload['annual_fee']['claim_id'] = 'invented-claim'
        await tool.on_invoke_tool(ToolContext(context=trace, tool_name=tool.name, tool_call_id='test-call', tool_arguments=json.dumps(payload)), json.dumps(payload))
        self.assertEqual(trace.tool_calls[-1]['result']['status'], 'insufficient_evidence')

    async def test_calculator_rejects_semantic_rate_mismatch(self):
        from scripts.researchers import ISSUER, ResearchTrace, build_researcher_agents
        agent = build_researcher_agents('test-model', legacy_tools=True)[ISSUER]
        tool = next(tool for tool in agent.tools if tool.name == 'calculate_verified_rewards')
        claim = {
            'card_id': 'wf-active-cash', 'claim_id': 'claim-2pct',
            'evidence_id': 'claim-2pct', 'field': 'issuer_grounded_claim',
            'value': 'Unlimited 2% cash rewards on eligible purchases with no annual fee.',
            'source_url': 'https://www.wellsfargo.com/card',
            'source_date': '2026-11-10', 'status': 'issuer_grounded',
        }
        trace = ResearchTrace(tool_calls=[{
            'name': 'search_official_terms', 'args': {'card_id': 'wf-active-cash'},
            'result': {'evidence': [claim]},
        }])
        payload = {
            'card_id': 'wf-active-cash',
            'spending': dict(grocery=0, dining=0, travel=0, gas=0,
                             online_retail=0, transit=0, streaming=0, other=100),
            'annual_fee': {'value': 0, 'claim_id': 'claim-2pct'},
            'bonus_categories': {'value': [], 'claim_id': 'claim-2pct'},
            'capped_categories': {'value': [], 'claim_id': 'claim-2pct'},
            'rules': [{'field': 'other_rate_pct', 'value': 9, 'claim_id': 'claim-2pct'}],
        }
        arguments = json.dumps(payload)
        await tool.on_invoke_tool(
            ToolContext(context=trace, tool_name=tool.name,
                        tool_call_id='semantic-mismatch', tool_arguments=arguments),
            arguments,
        )
        result = trace.tool_calls[-1]['result']
        self.assertEqual(result['status'], 'semantic_mismatch')
        self.assertEqual(result['field'], 'other_rate_pct')

    async def test_calculator_accepts_markdown_zero_annual_fee_claim(self):
        from scripts.researchers import ISSUER, ResearchTrace, build_researcher_agents
        agent = build_researcher_agents('test-model', legacy_tools=True)[ISSUER]
        tool = next(tool for tool in agent.tools if tool.name == 'calculate_verified_rewards')
        claim = {
            'card_id': 'wf-active-cash', 'claim_id': 'markdown-zero-fee',
            'evidence_id': 'markdown-zero-fee', 'field': 'issuer_grounded_claim',
            'value': '**Annual Fee:** $0. Unlimited 2% cash rewards on all eligible purchases with no spending cap.',
            'source_url': 'https://www.wellsfargo.com/credit-cards/active-cash/',
            'source_date': '2026-11-10', 'status': 'issuer_grounded',
        }
        trace = ResearchTrace(tool_calls=[{
            'name': 'search_official_terms', 'args': {'card_id': 'wf-active-cash'},
            'result': {'evidence': [claim]},
        }])
        payload = {
            'card_id': 'wf-active-cash',
            'spending': dict(grocery=0, dining=0, travel=0, gas=0,
                             online_retail=0, transit=0, streaming=0, other=100),
            'annual_fee': {'value': 0, 'claim_id': 'markdown-zero-fee'},
            'bonus_categories': {'value': [], 'claim_id': 'markdown-zero-fee'},
            'capped_categories': {'value': [], 'claim_id': 'markdown-zero-fee'},
            'rules': [{'field': 'other_rate_pct', 'value': 2, 'claim_id': 'markdown-zero-fee'}],
        }
        arguments = json.dumps(payload)
        await tool.on_invoke_tool(
            ToolContext(context=trace, tool_name=tool.name,
                        tool_call_id='markdown-zero-fee', tool_arguments=arguments),
            arguments,
        )
        self.assertEqual(trace.tool_calls[-1]['result']['status'], 'estimated')

    async def test_calculator_accepts_dated_issuer_snapshot_claim_ids(self):
        from scripts.researchers import ISSUER, ResearchTrace, build_researcher_agents
        agent = build_researcher_agents('test-model', legacy_tools=True)[ISSUER]
        tool = next(tool for tool in agent.tools if tool.name == 'calculate_verified_rewards')
        issuer = verify_issuer_terms('blue-cash-preferred')
        evidence = {item['field']: item for item in issuer['evidence']}
        self.assertTrue(all(item.get('claim_id') for item in issuer['evidence']))
        trace = ResearchTrace(tool_calls=[{
            'name': 'lookup_issuer_terms',
            'args': {'card_id': 'blue-cash-preferred', 'section': 'all'},
            'result': issuer,
        }])
        payload = {
            'card_id': 'blue-cash-preferred',
            # The calculator accepts monthly spending and annualizes internally.
            'spending': dict(grocery=800, dining=0, travel=0, gas=0,
                             online_retail=0, transit=0, streaming=0, other=0),
            'annual_fee': {'value': 95, 'claim_id': evidence['annual_fee_usd']['claim_id']},
            'bonus_categories': {
                'value': evidence['bonus_categories']['value'],
                'claim_id': evidence['bonus_categories']['claim_id'],
            },
            'capped_categories': {
                'value': evidence['capped_categories']['value'],
                'claim_id': evidence['capped_categories']['claim_id'],
            },
            'rules': [
                {'field': field, 'value': evidence[field]['value'],
                 'claim_id': evidence[field]['claim_id']}
                for field in ('grocery_rate_pct', 'grocery_cap_usd',
                              'grocery_over_cap_rate_pct', 'grocery_cap_period')
            ],
        }
        arguments = json.dumps(payload)
        await tool.on_invoke_tool(
            ToolContext(context=trace, tool_name=tool.name,
                        tool_call_id='snapshot-calculation', tool_arguments=arguments),
            arguments,
        )
        result = trace.tool_calls[-1]['result']
        self.assertEqual(result['status'], 'estimated')
        self.assertEqual(result['net_cash_back_usd'], 301)
