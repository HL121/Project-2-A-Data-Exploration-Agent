"""Regression tests for the Issuer issues seen in the Blue Cash Preferred vs Everyday run."""

import json
import unittest

from agents.tool_context import ToolContext

from scripts.researchers import (
    ISSUER,
    ResearchTrace,
    annualized_spending_categories,
    build_researcher_agents,
    normalize_cap_period,
    quote_states_intro_fee_only,
)

RATE = "Earn 6% cash back on eligible purchases at U.S. supermarkets"
CAP = ("This rate applies to up to $6,000 per year in purchases (after which the rate "
       "drops to 1% for the remainder of that year)")
FEE = "$0 intro annual fee for the first year, then $95."
TASK = ("Verify official current terms. The user's monthly spending is $800 on groceries "
        "($9,600/year) and $0 on other categories.")


def tool(name):
    return next(item for item in build_researcher_agents("test-model")[ISSUER].tools if item.name == name)


async def invoke(name, trace, arguments):
    payload = json.dumps(arguments)
    ctx = ToolContext(context=trace, tool_name=name, tool_call_id=name, tool_arguments=payload)
    return await tool(name).on_invoke_tool(ctx, payload)


def search_trace(tasks=()):
    claims = {"rate": RATE, "cap": CAP, "fee": FEE}
    return ResearchTrace(tasks=list(tasks), tool_calls=[{
        "name": "search_official_terms", "args": {"card_id": "blue-cash-preferred"},
        "result": {"evidence": [{
            "card_id": "blue-cash-preferred", "claim_id": claim_id, "evidence_id": claim_id,
            "field": "issuer_grounded_claim", "value": text, "status": "issuer_grounded",
            "source_url": "https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/",
        } for claim_id, text in claims.items()]},
    }])


def term(field, value, claim_id, quote):
    return {"field": field, "value": value, "claim_id": claim_id, "quote": quote}


def spending(**monthly):
    categories = ("grocery", "dining", "travel", "gas", "online_retail", "transit", "streaming", "other")
    return {category: monthly.get(category, 0) for category in categories}


class CapPeriodTests(unittest.TestCase):
    def test_model_wordings_map_to_supported_periods(self):
        self.assertEqual(normalize_cap_period("annual"), "annual")
        self.assertEqual(normalize_cap_period("per year"), "annual")
        self.assertEqual(normalize_cap_period("Calendar Year"), "calendar_year")
        self.assertEqual(normalize_cap_period("calendar-year"), "calendar_year")
        self.assertEqual(normalize_cap_period("quarterly"), "calendar_quarter")
        self.assertIsNone(normalize_cap_period("whenever"))


class IntroFeeTests(unittest.TestCase):
    def test_intro_fee_is_not_the_ongoing_fee(self):
        self.assertTrue(quote_states_intro_fee_only(FEE, 0))
        self.assertFalse(quote_states_intro_fee_only(FEE, 95))
        self.assertTrue(quote_states_intro_fee_only("$0 introductory annual fee the first year.", 0))

    def test_quotes_without_an_intro_fee_are_unaffected(self):
        self.assertFalse(quote_states_intro_fee_only("Annual fee: $0.", 0))
        self.assertFalse(quote_states_intro_fee_only("No annual fee. 0% intro APR for 15 months.", 0))
        self.assertFalse(quote_states_intro_fee_only("No annual fee, 0% intro APR for 15 months.", 0))


class AnnualizedSpendingTests(unittest.TestCase):
    def test_annual_total_passed_as_monthly_is_flagged(self):
        self.assertEqual(annualized_spending_categories(spending(grocery=9600), [TASK]), ["grocery"])
        self.assertEqual(annualized_spending_categories(spending(grocery=800), [TASK]), [])

    def test_unstated_twelvefold_amount_is_flagged(self):
        task = "The user spends $500 per month on dining."
        self.assertEqual(annualized_spending_categories(spending(dining=6000), [task]), ["dining"])

    def test_monthly_amount_that_happens_to_be_twelve_times_another_is_allowed(self):
        task = "Monthly spending: $100 on groceries and $1,200 on everything else."
        self.assertEqual(annualized_spending_categories(spending(grocery=100, other=1200), [task]), [])

    def test_no_amounts_in_task_flags_nothing(self):
        self.assertEqual(annualized_spending_categories(spending(grocery=9600), ["Verify terms."]), [])


class BlueCashPreferredRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def register(self, trace, terms):
        return await invoke("register_issuer_terms", trace, {"card_id": "blue-cash-preferred", "terms": terms})

    async def test_annual_cap_period_is_registered(self):
        result = await self.register(search_trace(), [term("grocery_cap_period", "annual", "cap", CAP)])
        self.assertEqual(result["status"], "registered")
        self.assertEqual(result["evidence"][0]["value"], "annual")

    async def test_intro_fee_cannot_be_registered_as_ongoing_fee(self):
        trace = search_trace()
        rejected = await self.register(trace, [term("annual_fee_usd", 0, "fee", FEE)])
        self.assertEqual(rejected["extraction_errors"][0]["error"], "intro_fee_not_ongoing")
        self.assertIn("first_year_annual_fee_usd", rejected["extraction_errors"][0]["next_action"])
        accepted = await self.register(trace, [
            term("annual_fee_usd", 95, "fee", FEE),
            term("first_year_annual_fee_usd", 0, "fee", FEE),
        ])
        self.assertEqual(accepted["status"], "registered")
        self.assertEqual({item["field"]: item["value"] for item in accepted["evidence"]},
                         {"annual_fee_usd": 95, "first_year_annual_fee_usd": 0})

    async def registered_trace(self, tasks=()):
        trace = search_trace(tasks)
        result = await self.register(trace, [
            term("grocery_rate_pct", 6, "rate", RATE),
            term("grocery_cap_status", "capped", "cap", CAP),
            term("grocery_cap_usd", 6000, "cap", CAP),
            term("grocery_cap_period", "annual", "cap", CAP),
            term("grocery_over_cap_rate_pct", 1, "cap", CAP),
            term("annual_fee_usd", 95, "fee", FEE),
        ])
        self.assertEqual(result["status"], "registered", result)
        return trace

    async def test_calculator_uses_annual_cap_and_ongoing_fee(self):
        trace = await self.registered_trace([TASK])
        result = await invoke("calculate_from_terms", trace, {
            "card_id": "blue-cash-preferred", "spending": spending(grocery=800), "assume_uncapped": [],
        })
        self.assertEqual(result["status"], "estimated")
        self.assertEqual(result["gross_cash_back_usd"], 396)
        self.assertEqual(result["annual_fee_usd"], 95)
        self.assertEqual(result["net_cash_back_usd"], 301)
        self.assertTrue(result["notes"])
        self.assertEqual(result["assumptions"], [])

    async def test_calculator_rejects_annual_total_as_monthly_spending(self):
        trace = await self.registered_trace([TASK])
        result = await invoke("calculate_from_terms", trace, {
            "card_id": "blue-cash-preferred", "spending": spending(grocery=9600), "assume_uncapped": [],
        })
        self.assertEqual(result["status"], "annualized_spending_suspected")
        self.assertEqual(result["categories"], ["grocery"])
        self.assertNotIn("net_cash_back_usd", result)
        self.assertIn("MONTHLY", result["retry_action"])

    async def trace_without_cap_status(self):
        trace = search_trace([TASK])
        result = await self.register(trace, [
            term("grocery_rate_pct", 6, "rate", RATE),
            term("grocery_cap_usd", 6000, "cap", CAP),
            term("grocery_cap_period", "annual", "cap", CAP),
            term("grocery_over_cap_rate_pct", 1, "cap", CAP),
            term("annual_fee_usd", 95, "fee", FEE),
        ])
        self.assertEqual(result["status"], "registered", result)
        return trace

    async def test_registered_cap_amount_implies_capped_status(self):
        trace = await self.trace_without_cap_status()
        result = await invoke("calculate_from_terms", trace, {
            "card_id": "blue-cash-preferred", "spending": spending(grocery=800), "assume_uncapped": [],
        })
        self.assertEqual(result["status"], "estimated")
        self.assertEqual(result["net_cash_back_usd"], 301)
        self.assertNotIn("grocery_cap_status", result["required_fields"])
        self.assertTrue(any("inferred as capped" in note for note in result["notes"]))

    async def test_assumed_uncapped_cannot_override_a_registered_cap(self):
        trace = await self.trace_without_cap_status()
        result = await invoke("calculate_from_terms", trace, {
            "card_id": "blue-cash-preferred", "spending": spending(grocery=800), "assume_uncapped": ["grocery"],
        })
        self.assertEqual(result["status"], "estimated")
        self.assertEqual(result["net_cash_back_usd"], 301)  # not 481 = 9,600 x 6% - 95
        self.assertEqual(result["assumptions"], [])

    async def test_unknown_cap_without_cap_amount_is_still_a_gap(self):
        trace = search_trace([TASK])
        await self.register(trace, [
            term("grocery_rate_pct", 6, "rate", RATE),
            term("annual_fee_usd", 95, "fee", FEE),
        ])
        result = await invoke("calculate_from_terms", trace, {
            "card_id": "blue-cash-preferred", "spending": spending(grocery=800), "assume_uncapped": [],
        })
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("grocery_cap_status", result["missing_fields"])

    async def test_missing_issuer_fee_falls_back_to_dated_catalog_fee(self):
        trace = search_trace([TASK])
        await self.register(trace, [
            term("grocery_rate_pct", 6, "rate", RATE),
            term("grocery_cap_usd", 6000, "cap", CAP),
            term("grocery_cap_period", "annual", "cap", CAP),
            term("grocery_over_cap_rate_pct", 1, "cap", CAP),
        ])
        result = await invoke("calculate_from_terms", trace, {
            "card_id": "blue-cash-preferred", "spending": spending(grocery=800), "assume_uncapped": [],
        })
        self.assertEqual(result["status"], "estimated")
        self.assertEqual(result["annual_fee_usd"], 95)
        self.assertEqual(result["net_cash_back_usd"], 301)
        fee = next(item for item in result["evidence"] if item["field"] == "annual_fee_usd")
        self.assertEqual(fee["status"], "catalog_snapshot")
        self.assertIn("67.credit", fee["source_url"])
        self.assertTrue(any("67.CREDIT catalog" in note for note in result["notes"]))

    async def test_issuer_fee_takes_priority_over_catalog_fee(self):
        trace = await self.trace_without_cap_status()
        result = await invoke("calculate_from_terms", trace, {
            "card_id": "blue-cash-preferred", "spending": spending(grocery=800), "assume_uncapped": [],
        })
        fee = next(item for item in result["evidence"] if item["field"] == "annual_fee_usd")
        self.assertEqual(fee["status"], "issuer_grounded")
        self.assertFalse(any("67.CREDIT catalog" in note for note in result.get("notes", [])))

    def test_calculator_schema_says_monthly(self):
        schema = json.dumps(tool("calculate_from_terms").params_json_schema)
        description = tool("calculate_from_terms").description
        self.assertIn("MONTHLY", description + schema)
        self.assertIn("grocery=800", description)


if __name__ == "__main__":
    unittest.main()
