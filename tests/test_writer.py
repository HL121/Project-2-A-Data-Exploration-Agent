import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from agents import ModelSettings

from scripts.researchers import COSTS, ISSUER
from scripts.writer import _prepare_writer_input, generate_report


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


class GenerateReportTests(unittest.IsolatedAsyncioTestCase):
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
        })
        self.assertNotIn("research_results", prompt)


if __name__ == "__main__":
    unittest.main()
