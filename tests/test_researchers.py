import tempfile
import unittest
from pathlib import Path

from issuer_live import product_excerpt
from researchers import (
    CARD_IDS,
    CFPB_FILE,
    check_catalog_welcome_offer,
    check_merchant_exclusion,
    estimate_annual_grocery_rewards,
    estimate_historical_costs,
    estimate_spending_rewards,
    research_costs,
    research_rewards,
    verify_issuer_terms,
)

PREFERRED, EVERYDAY = CARD_IDS


class ResearcherTests(unittest.TestCase):
    def test_rewards_and_issuer_keep_separate_sources(self):
        rewards = research_rewards(PREFERRED)["evidence"]
        issuer = verify_issuer_terms(PREFERRED)["evidence"]
        self.assertEqual({item["status"] for item in rewards}, {"catalog_snapshot"})
        self.assertEqual({item["status"] for item in issuer}, {"issuer_snapshot"})
        self.assertNotEqual(rewards[0]["source_url"], issuer[0]["source_url"])

    def test_cfpb_costs_preserve_blank_fee_and_percent_units(self):
        preferred = research_costs(PREFERRED)["evidence"]
        self.assertTrue(all(item["source_locator"] == "TCCP Survey, row 34" for item in preferred))
        self.assertEqual({item["status"] for item in preferred}, {"cfpb_survey"})
        self.assertIn(("annual_fee_usd", 95), [(item["field"], item["value"]) for item in preferred])
        everyday = research_costs(EVERYDAY)
        values = {item["field"]: item["value"] for item in everyday["evidence"]}
        self.assertNotIn("annual_fee_usd", values)
        self.assertEqual(values["purchase_apr_median_pct"], 28.49)
        self.assertEqual(values["foreign_transaction_fee_pct"], 2.7)
        self.assertTrue(any("blank" in gap for gap in everyday["gaps"]))

    def test_missing_or_ambiguous_cfpb_data_uses_labeled_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            survey = Path(directory) / "survey.csv"
            missing = research_costs(PREFERRED, survey)
            lines = CFPB_FILE.read_text(encoding="utf-8").splitlines()
            survey.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
            ambiguous = research_costs(PREFERRED, survey)
        for result in (missing, ambiguous):
            self.assertEqual({item["status"] for item in result["evidence"]}, {"historical_agreement"})
        self.assertIn("found 2", ambiguous["gaps"][0])

    def test_unknown_card_returns_no_evidence(self):
        for research in (research_rewards, research_costs, verify_issuer_terms):
            result = research("unknown-card")
            self.assertEqual(result["evidence"], [])
            self.assertTrue(result["gaps"])

    def test_ongoing_rewards_around_cap(self):
        cases = [(PREFERRED, 800, 301), (EVERYDAY, 800, 216),
                 (PREFERRED, 200, 49), (EVERYDAY, 200, 72), (PREFERRED, 500, 265)]
        for card_id, spend, expected in cases:
            with self.subTest(card_id=card_id, spend=spend):
                issuer = verify_issuer_terms(card_id)["evidence"]
                result = estimate_annual_grocery_rewards(card_id, spend, issuer)
                self.assertEqual(result["net_cash_back_usd"], expected)

    def test_targeted_terms_lookup_fills_calculation_gap(self):
        product = verify_issuer_terms(PREFERRED, "product")["evidence"]
        terms = verify_issuer_terms(PREFERRED, "terms")["evidence"]
        partial = estimate_annual_grocery_rewards(PREFERRED, 800, product)
        self.assertEqual(partial["missing_fields"], ["grocery_cap_period"])
        combined = estimate_annual_grocery_rewards(PREFERRED, 800, product + terms)
        self.assertEqual(combined["net_cash_back_usd"], 301)
        exclusions = next(item["value"] for item in terms if item["field"] == "grocery_exclusions")
        self.assertIn("warehouse clubs", exclusions)

    def test_conflicting_issuer_terms_are_reported(self):
        issuer = verify_issuer_terms(PREFERRED)["evidence"]
        annual_fee = next(item for item in issuer if item["field"] == "annual_fee_usd")
        result = estimate_annual_grocery_rewards(PREFERRED, 800, [*issuer, {**annual_fee, "value": 0}])
        self.assertEqual(result["status"], "conflicting_evidence")
        self.assertEqual(result["conflicting_fields"], ["annual_fee_usd"])

    def test_invalid_spend_is_rejected(self):
        for amount in ("-1", "Infinity", "NaN", "unknown"):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                estimate_annual_grocery_rewards(PREFERRED, amount, [])

    def test_independent_category_caps_and_base_rate(self):
        evidence = verify_issuer_terms(EVERYDAY)["evidence"]
        result = estimate_spending_rewards(EVERYDAY, {"grocery": 800, "gas": 700, "online_retail": 800,
                                                     "transit": 100}, evidence)
        amounts = {item["category"]: item["cash_back_usd"] for item in result["categories"]}
        self.assertEqual(amounts, {"grocery": 216, "gas": 204, "online_retail": 216, "transit": 12})
        self.assertEqual(result["net_cash_back_usd"], 648)

    def test_uncapped_category_and_annual_fee(self):
        evidence = verify_issuer_terms(PREFERRED)["evidence"]
        result = estimate_spending_rewards(PREFERRED, {"gas": 700, "streaming": 50}, evidence)
        self.assertEqual(result["gross_cash_back_usd"], 288)
        self.assertEqual(result["net_cash_back_usd"], 193)

    def test_targeted_category_evidence_required_for_cap_reset(self):
        evidence = verify_issuer_terms(EVERYDAY, "product")["evidence"]
        result = estimate_spending_rewards(EVERYDAY, {"online_retail": 800}, evidence)
        self.assertEqual(result["missing_fields"], ["online_retail_cap_period"])
        with self.assertRaises(ValueError):
            estimate_spending_rewards(EVERYDAY, {"shopping": 800}, evidence)

    def test_costs_distinguish_ongoing_interest_promotion_and_transfer_fee(self):
        evidence = research_costs(PREFERRED)["evidence"]
        values = {item["field"]: item["value"] for item in evidence}
        self.assertEqual(values["intro_purchase_apr_min_pct"], 0)
        self.assertEqual(values["intro_purchase_apr_median_months"], 12)
        result = estimate_historical_costs(PREFERRED, evidence, 1000, 1000, 100)
        self.assertEqual(result["foreign_transaction_fee_usd"], 27)
        self.assertEqual(result["balance_transfer_fee_usd"], 5)
        self.assertEqual(result["scenarios"]["median"]["estimated_incremental_cost_usd"], 304.4)
        no_balance = estimate_historical_costs(PREFERRED, evidence, annual_foreign_spend_usd=1000)
        self.assertIsNone(no_balance["scenarios"]["median"]["historical_apr_pct"])
        self.assertEqual(no_balance["scenarios"]["median"]["estimated_incremental_cost_usd"], 27)

    def test_cost_scenario_does_not_use_margins_as_apr(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = research_costs(PREFERRED, Path(directory) / "missing.csv")["evidence"]
        self.assertEqual(estimate_historical_costs(PREFERRED, evidence, 1000)["status"], "insufficient_evidence")

    def test_offer_threshold_and_snapshot_are_separate_from_ongoing_rewards(self):
        result = check_catalog_welcome_offer(PREFERRED, 2600)
        self.assertFalse(result["threshold_met_in_scenario"])
        self.assertEqual(result["remaining_spend_usd"], 400)
        self.assertEqual(result["window_months"], 6)
        self.assertTrue(check_catalog_welcome_offer(EVERYDAY, 2600)["threshold_met_in_scenario"])

    def test_merchant_exclusion_only_claims_what_terms_establish(self):
        exclusion = check_merchant_exclusion(PREFERRED, "grocery", "warehouse club")
        self.assertEqual(exclusion["status"], "excluded_from_category_bonus")
        self.assertEqual(check_merchant_exclusion(EVERYDAY, "online_retail", "restaurant")["status"],
                         "excluded_from_category_bonus")
        self.assertEqual(check_merchant_exclusion(PREFERRED, "online_retail", "retailer")["status"],
                         "no_category_bonus")
        self.assertEqual(check_merchant_exclusion(PREFERRED, "grocery", "supermarket")["status"],
                         "eligibility_unconfirmed")

    def test_live_excerpt_requires_correct_product_and_visible_relevant_terms(self):
        html = """
        <title>Blue Cash Everyday Card | American Express</title>
        <script>Supermarkets 95%</script>
        <h1>Blue Cash Everyday<sup>®</sup> Card</h1>
        <p>Earn <strong>3%</strong> at U.S. supermarkets, up to $6,000 per year, then 1%.</p>
        <p>The card has no annual fee.</p>
        """
        excerpt = product_excerpt(html, "Blue Cash Everyday", "grocery")
        self.assertIn("3% at U.S. supermarkets", excerpt)
        self.assertIn("$6,000", excerpt)
        self.assertNotIn("95%", excerpt)
        self.assertIn("no annual fee", product_excerpt(html, "Blue Cash Everyday", "annual_fee"))
        self.assertIsNone(product_excerpt(html, "Blue Cash Preferred", "grocery"))
        self.assertIsNone(product_excerpt(html, "Blue Cash Everyday", "transit"))
