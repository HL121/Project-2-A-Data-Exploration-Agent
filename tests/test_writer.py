import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from agents import ModelSettings

from scripts.researchers import COSTS, ISSUER
from scripts.writer import (
    _evidence_constraint_violations, _prepare_writer_input, _safe_incomplete_report,
    _violates_decision_gate,
    generate_report,
)


def grounded_claim(evidence_id, text="Annual fee: $95."):
    return {
        "card_id": "card-a", "field": "issuer_grounded_claim", "value": text,
        "claim_id": evidence_id, "evidence_id": evidence_id,
        "source_url": "https://issuer.example/card-a", "source_urls": ["https://issuer.example/card-a"],
        "source_date": "2026-11-11", "retrieved_at": "2026-11-11T10:00:00+00:00",
        "status": "issuer_grounded",
    }


def registered(field, value, claim_id="card-a:run:0"):
    return {
        **grounded_claim(claim_id), "field": field, "value": value, "quote": "Annual fee: $95.",
        "verification_method": "quoted_model_extraction", "evidence_id": f"{claim_id}:{field}",
    }


class WriterInputTests(unittest.TestCase):
    def issuer_results(self):
        """Two Issuer rounds whose results share one cumulative tool-call list."""
        shared_calls = [
            {"agent": ISSUER, "name": "search_official_terms",
             "args": {"card_id": "card-a", "task": "Verify fee"},
             "result": {"card_id": "card-a", "status": "grounded",
                        "evidence": [grounded_claim("card-a:run:0")],
                        "gaps": ["Published terms do not establish personal approval."],
                        "sources": [{"url": "https://other.example", "official": False}],
                        "web_search_queries": ["card a annual fee"],
                        "search_entry_point_html": ["<div>suggestions</div>"]}},
            {"agent": ISSUER, "name": "register_issuer_terms",
             "args": {"card_id": "card-a", "terms": []},
             "result": {"card_id": "card-a", "status": "partial",
                        "evidence": [registered("annual_fee_usd", 95)],
                        "extraction_errors": [
                            {"field": "grocery_rate_pct", "error": "numeric_mismatch",
                             "supporting_claim_text": "...", "next_action": "Correct only this field."},
                            {"field": "grocery_cap_status", "error": "quote_or_claim_mismatch",
                             "next_action": "Copy an exact quote."},
                        ]}},
            {"agent": ISSUER, "name": "calculate_from_terms",
             "args": {"card_id": "card-a", "spending": {"grocery": 800}, "assume_uncapped": []},
             "result": {"card_id": "card-a", "status": "insufficient_evidence",
                        "missing_fields": ["grocery_rate_pct"],
                        "error_kind": "evidence_or_parameter",
                        "retry_action": "Correct the named field."}},
        ]
        round_one = {"agent": ISSUER, "round": 1, "card_ids": ["card-a"], "task": "Verify",
                     "status": "incomplete", "summary": "{}", "attempts": 1,
                     "findings": [call["result"] for call in shared_calls],
                     "tool_calls": shared_calls}
        shared_calls.extend([
            {"agent": ISSUER, "name": "register_issuer_terms",
             "args": {"card_id": "card-a", "terms": ["repair"]},
             "result": {"card_id": "card-a", "status": "registered",
                        "evidence": [registered("grocery_rate_pct", 3)], "extraction_errors": []}},
            {"agent": ISSUER, "name": "calculate_from_terms",
             "args": {"card_id": "card-a", "spending": {"grocery": 800}, "assume_uncapped": []},
             "result": {"status": "duplicate_blocked", "error_kind": "no_progress",
                        "retry_action": "Use the previous result."}},
            {"agent": ISSUER, "name": "calculate_from_terms",
             "args": {"card_id": "card-a", "spending": {"grocery": 800}, "assume_uncapped": ["grocery"]},
             "result": {"card_id": "card-a", "status": "conditional_estimate",
                        "categories": [{"category": "grocery", "status": "estimated",
                                        "cash_back_usd": 288.0}],
                        "assumptions": ["Assume no reward cap for grocery; not established by evidence."],
                        "gross_cash_back_usd": 288.0, "annual_fee_usd": 95.0,
                        "net_cash_back_usd": 193.0,
                        "evidence": [registered("annual_fee_usd", 95),
                                     registered("grocery_rate_pct", 3)]}},
        ])
        round_two = {**round_one, "round": 2, "task": "Repair grocery rate",
                     "findings": [call["result"] for call in shared_calls]}
        return [round_one, round_two]

    def prepare(self, results=None, assessment=None):
        return _prepare_writer_input(
            "Which card?", results if results is not None else self.issuer_results(),
            assessment if assessment is not None else {"research_status": "completed"},
        )

    def test_shared_tool_calls_are_deduplicated(self):
        payload = self.prepare()
        evidence_ids = [item["evidence_id"] for item in payload["evidence"]]
        self.assertEqual(len(evidence_ids), len(set(evidence_ids)))
        self.assertEqual(set(evidence_ids), {
            "card-a:run:0", "card-a:run:0:annual_fee_usd", "card-a:run:0:grocery_rate_pct",
        })
        searches = [item for item in payload["tool_results"] if item["tool"] == "search_official_terms"]
        self.assertEqual(len(searches), 1)

    def test_raw_traces_and_researcher_guidance_are_removed(self):
        payload = self.prepare()
        text = json.dumps(payload)
        for removed in ("retry_action", "error_kind", "next_action", "supporting_claim_text",
                        "search_entry_point_html", "web_search_queries", "duplicate_blocked",
                        "other.example", '"findings"', '"tool_calls"', '"attempts"'):
            self.assertNotIn(removed, text)
        search = payload["tool_results"][0]
        self.assertEqual(search["status"], "grounded")
        self.assertEqual(search["evidence_ids"], ["card-a:run:0"])
        self.assertIn("gaps", search)

    def test_latest_calculation_per_card_keeps_inputs_and_assumptions(self):
        calculation = self.prepare()["calculations_by_card"]["card-a"]
        self.assertEqual(calculation["status"], "conditional_estimate")
        self.assertEqual(calculation["inputs"], {"spending": {"grocery": 800},
                                                 "assume_uncapped": ["grocery"]})
        self.assertEqual(calculation["assumptions"],
                         ["Assume no reward cap for grocery; not established by evidence."])
        self.assertEqual(calculation["evidence_ids"],
                         ["card-a:run:0:annual_fee_usd", "card-a:run:0:grocery_rate_pct"])
        self.assertNotIn("evidence", calculation)

    def test_only_unrepaired_extraction_failures_are_reported(self):
        payload = self.prepare()
        self.assertEqual(payload["unresolved_extraction_failures"], {
            "card-a": [{"field": "grocery_cap_status", "reason": "quote_or_claim_mismatch"}],
        })
        self.assertFalse(any(item["tool"] == "register_issuer_terms"
                             for item in payload["tool_results"]))

    def test_verified_facts_gain_sources_without_changing_lead_assessment(self):
        assessment = {
            "research_status": "completed",
            "verified_facts": [{"statement": "card-a: annual_fee_usd=95",
                                "evidence_ids": ["card-a:run:0:annual_fee_usd"]}],
            "resolved_conflicts": [],
            "unresolved_gaps": ["Grocery cap unknown."],
            "issuer_field_coverage": {"card-a": {"missing_fields": []}},
            "structured_quality_issues": [],
            "research_state": {"tool_budgets": {}},
        }
        original = copy.deepcopy(assessment)
        writer_assessment = self.prepare(assessment=assessment)["lead_assessment"]
        self.assertEqual(assessment, original)
        self.assertNotIn("research_state", writer_assessment)
        self.assertNotIn("structured_quality_issues", writer_assessment)
        self.assertEqual(writer_assessment["unresolved_gaps"], ["Grocery cap unknown."])
        self.assertEqual(writer_assessment["verified_facts"][0]["sources"], [{
            "evidence_id": "card-a:run:0:annual_fee_usd",
            "source_url": "https://issuer.example/card-a",
            "source_date": "2026-11-11",
            "status": "issuer_grounded",
        }])

    def test_researcher_inputs_are_not_modified(self):
        results = self.issuer_results()
        original = copy.deepcopy(results)
        self.prepare(results=results)
        self.assertEqual(results, original)

    def test_failed_researcher_report_keeps_error(self):
        failed = {"agent": COSTS, "round": 1, "card_ids": ["card-a"], "task": "Costs",
                  "status": "failed", "attempts": 2, "error_type": "TimeoutError",
                  "error": "timed out", "attempt_errors": [{}], "findings": [], "tool_calls": [],
                  "unresearched_card_ids": ["card-a"]}
        report = self.prepare(results=[failed])["researcher_reports"][0]
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_type"], "TimeoutError")
        self.assertEqual(report["unresearched_card_ids"], ["card-a"])
        self.assertNotIn("attempt_errors", report)

    def test_cost_scenario_keeps_its_inputs(self):
        cost_call = {"agent": COSTS, "name": "model_historical_costs",
                     "args": {"card_id": "card-a", "average_interest_bearing_balance_usd": 1000,
                              "annual_foreign_spend_usd": 0, "balance_transfer_usd": 0},
                     "result": {"card_id": "card-a", "status": "historical_scenario",
                                "scenarios": {"median": {"historical_apr_pct": 24.0}},
                                "evidence": []}}
        result = {"agent": COSTS, "round": 1, "card_ids": ["card-a"], "task": "Costs",
                  "status": "completed", "summary": "ok", "tool_calls": [cost_call]}
        outcome = self.prepare(results=[result])["tool_results"][0]
        self.assertEqual(outcome["inputs"]["average_interest_bearing_balance_usd"], 1000)
        self.assertEqual(outcome["scenarios"]["median"]["historical_apr_pct"], 24.0)

    def test_card_reportability_preserves_unknowns(self):
        payload = self.prepare()
        card = payload["card_reportability"]["card-a"]
        self.assertEqual(card["cap_status"], "unknown")
        self.assertFalse(card["point_value_verified"])
        self.assertFalse(card["reportable_annual_total"])
        self.assertTrue(card["official_terms_verified"])

    def test_partial_calculations_get_scoped_comparison_readiness(self):
        results = []
        for card_id, net in (("card-a", 200), ("card-b", 180)):
            results.append({
                "agent": ISSUER, "round": 1, "card_ids": [card_id],
                "task": "Calculate verified rewards", "status": "completed",
                "tool_calls": [{
                    "name": "calculate_from_terms", "args": {"card_id": card_id},
                    "result": {"card_id": card_id, "status": "estimated",
                               "net_cash_back_usd": net},
                }],
            })
        results.append({
            "agent": ISSUER, "round": 1, "card_ids": ["card-c"],
            "task": "Calculate verified rewards", "status": "incomplete", "tool_calls": [],
        })
        readiness = self.prepare(
            results=results, assessment={"research_status": "incomplete"},
        )["decision_readiness"]
        self.assertEqual(readiness["conclusion_scope"], "verified_candidates_only")
        self.assertEqual(readiness["comparison_eligible_cards"], ["card-a", "card-b"])
        self.assertEqual(readiness["excluded_from_ranking_cards"], ["card-c"])

    def test_evidence_gate_rejects_unknown_cap_point_value_and_manual_total(self):
        payload = self.prepare()
        report = (
            "Card A has no cap and is unlimited.\n"
            "Card A has a statement credit value of 0.6 cents per point.\n"
            "Card A estimated annual rewards: $180.\n"
            "Cards in the broader market often feature higher rates."
        )
        violations = _evidence_constraint_violations(report, payload)
        self.assertTrue(any("unknown cap" in item for item in violations))
        self.assertTrue(any("point redemption" in item for item in violations))
        self.assertTrue(any("annual total" in item for item in violations))
        self.assertTrue(any("market knowledge" in item for item in violations))

    def test_evidence_gate_matches_common_card_display_names(self):
        payload = {"card_reportability": {
            "wf-active-cash": {
                "name": "Wells Fargo Active Cash® Card", "cap_status": "unknown",
                "point_value_verified": False, "reportable_annual_total": False,
                "official_terms_verified": False,
            },
            "amex-everyday": {
                "name": "Amex EveryDay® Credit Card", "cap_status": "capped",
                "point_value_verified": False, "reportable_annual_total": False,
                "official_terms_verified": False,
            },
        }}
        report = (
            "| Wells Fargo Active Cash | 2% | None |\n"
            "The Amex EveryDay statement credit value is 0.6 cents per point."
        )
        violations = _evidence_constraint_violations(report, payload)
        self.assertTrue(any("wf-active-cash" in item and "unknown cap" in item for item in violations))
        self.assertTrue(any("amex-everyday" in item and "point redemption" in item for item in violations))

    def test_evidence_gate_rejects_internal_instruction_and_unknown_table_placeholder(self):
        payload = {"card_reportability": {}}
        report = (
            "Research is incomplete; report verified findings without declaring a definitive winner.\n"
            "| Grocery exclusions | N/A |\n"
        )
        violations = _evidence_constraint_violations(report, payload)
        self.assertTrue(any("internal instruction" in item for item in violations))
        self.assertTrue(any("N/A" in item for item in violations))

    def test_safe_fallback_is_scoped_and_user_facing(self):
        payload = {
            "calculations_by_card": {
                "savor": {
                    "status": "estimated", "gross_cash_back_usd": 432,
                    "annual_fee_usd": 0, "net_cash_back_usd": 432,
                    "inputs": {"spending": {
                        "dining": 700, "entertainment": 300, "grocery": 200,
                    }},
                    "categories": [
                        {"category": "dining", "status": "estimated", "cash_back_usd": 252},
                        {"category": "entertainment", "status": "estimated", "cash_back_usd": 108},
                        {"category": "grocery", "status": "estimated", "cash_back_usd": 72},
                    ],
                    "evidence_ids": ["savor-rate"],
                },
                "alliant": {
                    "status": "estimated", "gross_cash_back_usd": 216,
                    "annual_fee_usd": 0, "net_cash_back_usd": 216,
                    "inputs": {"spending": {"other": 1200}}, "categories": [],
                    "evidence_ids": ["alliant-rate"],
                },
            },
            "card_reportability": {
                "savor": {"name": "Capital One Savor"},
                "alliant": {"name": "Alliant Cashback Visa"},
                "altitude": {"name": "U.S. Bank Altitude Go"},
            },
            "decision_readiness": {"excluded_from_ranking_cards": ["altitude"]},
            "evidence": [
                {"evidence_id": "savor-rate", "source_url": "https://issuer.example/savor"},
                {"evidence_id": "alliant-rate", "source_url": "https://issuer.example/alliant"},
            ],
            "lead_assessment": {"verified_facts": [{"statement": "issuer_grounded_claim=raw"}]},
        }
        answer = _safe_incomplete_report(payload)
        self.assertIn("Among the verified cards", answer)
        self.assertIn("$432.00", answer)
        self.assertIn("Entertainment: **$108.00/year** from $300.00/month", answer)
        self.assertIn("What could change this comparison", answer)
        self.assertIn("shifting spending among them does not", answer)
        self.assertIn("It was excluded", answer)
        self.assertIn("https://issuer.example/savor", answer)
        self.assertNotIn("issuer_grounded_claim", answer)


class GenerateReportTests(unittest.IsolatedAsyncioTestCase):
    def test_incomplete_gate_allows_explicitly_negated_superlative(self):
        self.assertFalse(_violates_decision_gate(
            "The evidence cannot determine the highest-return card.",
            {"can_declare_winner": False},
        ))

    def test_incomplete_gate_allows_explicitly_scoped_verified_comparison(self):
        self.assertFalse(_violates_decision_gate(
            "Among the verified cards, Card A has the highest estimate; Card C was excluded.",
            {"can_declare_winner": False, "conclusion_scope": "verified_candidates_only"},
        ))
        self.assertTrue(_violates_decision_gate(
            "Card A has the highest estimate.",
            {"can_declare_winner": False, "conclusion_scope": "verified_candidates_only"},
        ))

    async def test_writer_receives_prepared_input(self):
        result = type("Result", (), {"final_output": "# Report"})()
        results = WriterInputTests().issuer_results()
        with patch("scripts.writer.Runner.run", new=AsyncMock(return_value=result)) as run:
            answer = await generate_report(
                "Which card?", results, {"research_status": "incomplete"},
                model="test-model", model_settings=ModelSettings(),
            )
        prompt = json.loads(run.await_args.args[1])
        self.assertEqual(answer, "# Report")
        self.assertEqual(set(prompt), {
            "user_input", "lead_assessment", "researcher_reports", "evidence",
            "calculations_by_card", "tool_results", "unresolved_extraction_failures",
            "decision_readiness", "card_reportability",
        })
        self.assertFalse(prompt["decision_readiness"]["can_declare_winner"])
        self.assertIn("card-a", prompt["decision_readiness"]["manual_calculation_prohibited_cards"])
        self.assertNotIn("research_results", prompt)

    async def test_incomplete_writer_claim_is_rewritten(self):
        first = type("Result", (), {"final_output": "Card A is the best card."})()
        second = type("Result", (), {"final_output": "Research is incomplete; Card A was verified."})()
        with patch("scripts.writer.Runner.run", new=AsyncMock(side_effect=[first, second])) as run:
            answer = await generate_report(
                "Which card?", [], {"research_status": "incomplete", "unresolved_gaps": ["Card B"]},
                model="test-model", model_settings=ModelSettings(),
            )
        self.assertEqual(answer, "Research is incomplete; Card A was verified.")
        self.assertEqual(run.await_count, 2)
        correction = json.loads(run.await_args_list[1].args[1])
        self.assertIn("source_material", correction)

    async def test_repeated_incomplete_writer_claim_uses_safe_fallback(self):
        bad = type("Result", (), {"final_output": "Card A is the highest and best."})()
        with patch("scripts.writer.Runner.run", new=AsyncMock(return_value=bad)):
            answer = await generate_report(
                "Which card?", [], {"research_status": "incomplete", "unresolved_gaps": ["Card B"]},
                model="test-model", model_settings=ModelSettings(),
            )
        self.assertIn("cannot support a ranking", answer)
        self.assertIn("Card B", answer)


if __name__ == "__main__":
    unittest.main()
