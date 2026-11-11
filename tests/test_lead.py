import asyncio
import json
import unittest
from unittest.mock import ANY, AsyncMock, patch

from agents import ModelSettings
from agents.extensions.models.litellm_model import LitellmModel
from agents.tool_context import ToolContext

from scripts.lead import (
    MAX_RESEARCH_ROUNDS,
    MAX_CONCURRENT_RESEARCHERS,
    MAX_TOTAL_RESEARCHER_CALLS,
    RateLimitFallbackLitellmModel,
    SETTINGS,
    ResearchRun,
    build_lead_agent,
    deep_research,
)
from scripts.researchers import COSTS, ISSUER, REWARDS
from scripts.writer import generate_report


class LeadOrchestrationTests(unittest.TestCase):
    def make_run(self):
        return ResearchRun(hooks=None, researcher_agents={})

    def test_round_cap_is_enforced_by_state(self):
        run = self.make_run()
        discovery = run.reserve_assignment(0, REWARDS, [], "Discover candidates")
        run.finish_assignment(discovery, succeeded=True)
        for researcher in (REWARDS, COSTS, ISSUER):
            assignment = run.reserve_assignment(
                1, researcher, ["wf-active-cash"], f"Round one {researcher}",
            )
            run.finish_assignment(assignment, succeeded=True)
        for round_number in range(2, MAX_RESEARCH_ROUNDS + 1):
            assignment = run.reserve_assignment(
                round_number, REWARDS, ["wf-active-cash"], f"Research round {round_number}",
            )
            run.finish_assignment(assignment, succeeded=True)
        with self.assertRaises(ValueError):
            run.reserve_assignment(MAX_RESEARCH_ROUNDS + 1, COSTS, ["wf-active-cash"], "Follow up")

    def test_same_researcher_can_run_distinct_tasks_in_one_round(self):
        run = self.make_run()
        run.reserve_assignment(1, REWARDS, ["wf-active-cash"], "Research ongoing rewards")
        run.reserve_assignment(1, REWARDS, ["wf-active-cash"], "Research welcome offer")
        self.assertEqual(len(run.assignments), 2)

    def test_identical_assignment_is_rejected_across_rounds(self):
        run = self.make_run()
        assignment = run.reserve_assignment(1, REWARDS, ["wf-active-cash"], "Research   ongoing rewards")
        run.finish_assignment(assignment, succeeded=True)
        with self.assertRaises(ValueError):
            run.reserve_assignment(2, REWARDS, ["wf-active-cash"], "research ongoing REWARDS")

    def test_total_researcher_call_limit_is_enforced(self):
        run = self.make_run()
        for index in range(MAX_TOTAL_RESEARCHER_CALLS):
            assignment = run.reserve_assignment(1, REWARDS, ["wf-active-cash"], f"Distinct task {index}")
            run.reserve_researcher_call(assignment)
            run.finish_assignment(assignment, succeeded=True)
        with self.assertRaises(ValueError):
            run.reserve_assignment(2, COSTS, ["wf-active-cash"], "One call too many")

    def test_formal_rounds_run_in_order(self):
        run = self.make_run()
        with self.assertRaisesRegex(ValueError, "cannot be skipped"):
            run.reserve_assignment(2, COSTS, ["wf-active-cash"], "Premature round two")
        round_one = run.reserve_assignment(1, REWARDS, ["wf-active-cash"], "Round one")
        with self.assertRaisesRegex(ValueError, "still running"):
            run.reserve_assignment(2, COSTS, ["wf-active-cash"], "Round two")
        run.finish_assignment(round_one, succeeded=True)
        run.reserve_assignment(2, COSTS, ["wf-active-cash"], "Round two")
        with self.assertRaisesRegex(ValueError, "cannot move backward"):
            run.reserve_assignment(1, COSTS, ["wf-active-cash"], "Late round one task")

    def test_multicard_round_one_requires_only_planned_researchers(self):
        run = self.make_run()
        cards = ["blue-cash-preferred", "blue-cash-everyday"]
        run.required_round_one_researchers = {ISSUER, COSTS}
        issuer = run.reserve_assignment(1, ISSUER, cards, "Verify current terms")
        run.finish_assignment(issuer, succeeded=True)
        self.assertEqual(set(run.missing_required_round_one_researchers()), {COSTS})
        with self.assertRaisesRegex(ValueError, "planned round-one research is incomplete"):
            run.reserve_assignment(2, ISSUER, cards, "Follow up")

        costs = run.reserve_assignment(1, COSTS, cards, "Compare costs")
        run.finish_assignment(costs, succeeded=True)
        self.assertEqual(run.missing_required_round_one_researchers(), [])
        run.reserve_assignment(2, ISSUER, cards, "Targeted follow up")

    def test_single_card_single_researcher_does_not_trigger_full_fanout(self):
        run = self.make_run()
        assignment = run.reserve_assignment(
            1, ISSUER, ["wf-active-cash"], "Verify the annual fee",
        )
        run.finish_assignment(assignment, succeeded=True)
        self.assertEqual(run.missing_required_round_one_researchers(), [])

    def test_successful_discovery_can_run_only_once(self):
        run = self.make_run()
        discovery = run.reserve_assignment(0, REWARDS, [], "Discover candidates")
        run.finish_assignment(discovery, succeeded=True)
        with self.assertRaisesRegex(ValueError, "already completed successfully"):
            run.reserve_assignment(0, REWARDS, [], "Try a different discovery")

    def test_lead_has_selection_and_three_delegation_tools(self):
        names = {tool.name for tool in build_lead_agent("test-model").tools}
        self.assertEqual(
            names,
            {
                "find_cards",
                "discover_candidates",
                "delegate_rewards",
                "delegate_costs",
                "delegate_issuer_verification",
                "run_round_one",
                "save_research_assessment",
            },
        )

    def test_vertex_429_uses_runner_backoff_instead_of_immediate_retries(self):
        self.assertEqual(SETTINGS.extra_args["num_retries"], 0)
        self.assertEqual(SETTINGS.retry.max_retries, 4)
        self.assertEqual(SETTINGS.retry.backoff.initial_delay, 3)
        self.assertEqual(SETTINGS.retry.backoff.max_delay, 20)
        self.assertEqual(SETTINGS.retry.backoff.multiplier, 2)
        self.assertIsNotNone(SETTINGS.retry.policy)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_multicard_round_one_starts_all_three_researchers_concurrently(self):
        agents = {REWARDS: object(), COSTS: object(), ISSUER: object()}
        names_by_agent = {agent: name for name, agent in agents.items()}
        state = ResearchRun(hooks=None, researcher_agents=agents)
        lead = build_lead_agent("test-model")
        tool = next(tool for tool in lead.tools if tool.name == "run_round_one")
        active = 0
        maximum_active = 0
        all_started = asyncio.Event()

        async def run_researcher(agent, *args, **kwargs):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 3:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=1)
            active -= 1
            return {
                "agent": names_by_agent[agent], "summary": "Done.", "findings": [],
                "tool_calls": [], "unresearched_card_ids": [], "quality_issues": [],
            }

        payload = {
            "card_ids": ["blue-cash-preferred", "blue-cash-everyday"],
            "rewards_task": "Compare ongoing rewards",
            "costs_task": "Compare historical costs",
            "issuer_task": "Verify current official terms",
        }
        arguments = json.dumps(payload)
        context = ToolContext(
            context=state, tool_name=tool.name,
            tool_call_id="round-one-fanout", tool_arguments=arguments,
        )
        with patch("scripts.lead.run_researcher", new=run_researcher):
            result = await tool.on_invoke_tool(context, arguments)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(maximum_active, 3)
        self.assertEqual({item["researcher"] for item in state.assignments}, set(agents))

    async def test_multicard_round_one_skips_empty_researcher_tasks(self):
        agents = {REWARDS: object(), COSTS: object(), ISSUER: object()}
        state = ResearchRun(hooks=None, researcher_agents=agents)
        lead = build_lead_agent("test-model")
        tool = next(tool for tool in lead.tools if tool.name == "run_round_one")
        arguments = json.dumps({
            "card_ids": ["blue-cash-preferred", "blue-cash-everyday"],
            "rewards_task": "",
            "costs_task": "",
            "issuer_task": "Verify current rewards and calculate returns",
        })
        context = ToolContext(context=state, tool_name=tool.name,
                              tool_call_id="selective-round-one", tool_arguments=arguments)
        result = {
            "agent": ISSUER, "summary": "Done.", "findings": [], "tool_calls": [],
            "unresearched_card_ids": [], "quality_issues": [],
        }
        with patch("scripts.lead.run_researcher", new=AsyncMock(return_value=result)) as run:
            outcome = await tool.on_invoke_tool(context, arguments)
        self.assertEqual(outcome["status"], "completed")
        run.assert_awaited_once()
        self.assertEqual({item["researcher"] for item in state.assignments}, {ISSUER})
        self.assertEqual(state.required_round_one_researchers, {ISSUER})

    async def test_direct_multicard_round_one_delegation_is_rejected(self):
        state = ResearchRun(hooks=None, researcher_agents={REWARDS: object()})
        lead = build_lead_agent("test-model")
        tool = next(tool for tool in lead.tools if tool.name == "delegate_rewards")
        arguments = json.dumps({
            "card_ids": ["blue-cash-preferred", "blue-cash-everyday"],
            "task": "Compare rewards", "round_number": 1,
        })
        context = ToolContext(context=state, tool_name=tool.name,
                              tool_call_id="direct-multicard", tool_arguments=arguments)
        result = await tool.on_invoke_tool(context, arguments)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("run_round_one", result["error"])

    async def test_calculator_task_cannot_be_sent_to_rewards(self):
        state = ResearchRun(hooks=None, researcher_agents={REWARDS: object()})
        lead = build_lead_agent("test-model")
        tool = next(tool for tool in lead.tools if tool.name == "delegate_rewards")
        arguments = json.dumps({
            "card_ids": ["wf-active-cash"],
            "task": "Calculate annual net rewards using verified current terms",
            "round_number": 1,
        })
        context = ToolContext(context=state, tool_name=tool.name,
                              tool_call_id="wrong-calculator-route", tool_arguments=arguments)
        with patch("scripts.lead.run_researcher", new=AsyncMock()) as run:
            result = await tool.on_invoke_tool(context, arguments)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("Issuer Verification", result["error"])
        run.assert_not_awaited()

    async def test_lead_falls_back_only_for_vertex_resource_exhaustion(self):
        calls = []

        async def get_response(model, *args, **kwargs):
            calls.append(model.model)
            if model.model == "primary-model":
                raise RuntimeError("429 RESOURCE_EXHAUSTED: Resource exhausted")
            return "fallback-response"

        model = RateLimitFallbackLitellmModel("primary-model", "fallback-model")
        with patch.object(LitellmModel, "get_response", new=get_response):
            response = await model.get_response()
        self.assertEqual(response, "fallback-response")
        self.assertEqual(calls, ["primary-model", "fallback-model"])

    async def test_lead_does_not_fallback_for_non_rate_limit_error(self):
        calls = []

        async def get_response(model, *args, **kwargs):
            calls.append(model.model)
            raise ValueError("invalid request")

        model = RateLimitFallbackLitellmModel("primary-model", "fallback-model")
        with patch.object(LitellmModel, "get_response", new=get_response):
            with self.assertRaisesRegex(ValueError, "invalid request"):
                await model.get_response()
        self.assertEqual(calls, ["primary-model"])

    async def test_researcher_concurrency_is_bounded(self):
        researcher = object()
        state = ResearchRun(hooks=None, researcher_agents={REWARDS: researcher})
        lead = build_lead_agent("test-model")
        tool = next(tool for tool in lead.tools if tool.name == "delegate_rewards")
        active = 0
        maximum_active = 0
        first_pair_started = asyncio.Event()

        async def run_researcher(*args, **kwargs):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            if active == MAX_CONCURRENT_RESEARCHERS:
                first_pair_started.set()
            await first_pair_started.wait()
            await asyncio.sleep(0)
            active -= 1
            return {
                "agent": REWARDS,
                "summary": "Done.",
                "findings": [],
                "tool_calls": [],
                "unresearched_card_ids": [],
                "quality_issues": [],
            }

        contexts = []
        for index in range(3):
            arguments = json.dumps({
                "card_ids": ["wf-active-cash"],
                "task": f"Distinct rewards task {index}",
                "round_number": 1,
            })
            contexts.append(ToolContext(
                context=state,
                tool_name=tool.name,
                tool_call_id=f"concurrency-{index}",
                tool_arguments=arguments,
            ))

        with patch("scripts.lead.run_researcher", new=run_researcher):
            await asyncio.gather(*(
                tool.on_invoke_tool(context, context.tool_arguments)
                for context in contexts
            ))
        self.assertEqual(maximum_active, MAX_CONCURRENT_RESEARCHERS)

    async def test_failed_researcher_call_is_retried_once(self):
        researcher = object()
        state = ResearchRun(hooks=None, researcher_agents={REWARDS: researcher})
        lead = build_lead_agent("test-model")
        tool = next(tool for tool in lead.tools if tool.name == "delegate_rewards")
        success = {
            "agent": REWARDS,
            "summary": "Recovered on retry.",
            "findings": [],
            "tool_calls": [],
            "unresearched_card_ids": [],
        }
        arguments = {
            "card_ids": ["wf-active-cash"],
            "task": "Research rewards",
            "round_number": 1,
        }
        context = ToolContext(
            context=state,
            tool_name=tool.name,
            tool_call_id="retry-test",
            tool_arguments=json.dumps(arguments),
        )
        with patch(
            "scripts.lead.run_researcher",
            new=AsyncMock(side_effect=[RuntimeError("temporary failure"), success]),
        ) as run:
            await tool.on_invoke_tool(context, context.tool_arguments)
        self.assertEqual(run.await_count, 2)
        self.assertEqual(state.researcher_call_count, 2)
        self.assertEqual(state.research_results[0]["attempts"], 2)
        self.assertEqual(state.research_results[0]["status"], "completed")

    async def test_lead_assessment_tool_saves_structured_handoff(self):
        state = ResearchRun(hooks=None, researcher_agents={})
        state.research_results.append({
            "agent": REWARDS,
            "tool_calls": [{
                "name": "lookup_rewards",
                "args": {"card_id": "wf-active-cash"},
                "result": {"evidence": [{
                    "evidence_id": "evidence-1",
                    "card_id": "wf-active-cash",
                    "field": "availability",
                    "value": "active",
                }]},
            }],
        })
        lead = build_lead_agent("test-model")
        tool = next(tool for tool in lead.tools if tool.name == "save_research_assessment")
        payload = {
            "research_status": "incomplete",
            "verified_facts": [{
                "statement": "The catalog identifies the card as active.",
                "evidence_ids": ["evidence-1"],
            }],
            "resolved_conflicts": ["Current issuer fee supersedes the historical survey fee."],
            "unresolved_gaps": ["Welcome bonus eligibility was not verified."],
        }
        context = ToolContext(
            context=state,
            tool_name=tool.name,
            tool_call_id="assessment-test",
            tool_arguments=json.dumps(payload),
        )
        await tool.on_invoke_tool(context, context.tool_arguments)
        self.assertEqual(state.lead_assessment["research_status"], payload["research_status"])
        self.assertEqual(state.lead_assessment["verified_facts"][0]["evidence_ids"], ["evidence-1"])
        self.assertIn("availability=active", state.lead_assessment["verified_facts"][0]["statement"])
        self.assertEqual(state.lead_assessment["resolved_conflicts"], payload["resolved_conflicts"])
        self.assertEqual(state.lead_assessment["unresolved_gaps"], payload["unresolved_gaps"])
        self.assertEqual(state.lead_assessment["issuer_field_coverage"], {})

    async def test_assessment_rejects_unknown_or_missing_evidence_ids(self):
        state = ResearchRun(hooks=None, researcher_agents={})
        lead = build_lead_agent("test-model")
        tool = next(tool for tool in lead.tools if tool.name == "save_research_assessment")
        for evidence_ids in ([], ["invented-evidence"]):
            payload = {
                "research_status": "incomplete",
                "verified_facts": [{
                    "statement": "Unsupported fact.",
                    "evidence_ids": evidence_ids,
                }],
                "resolved_conflicts": [],
                "unresolved_gaps": [],
            }
            arguments = json.dumps(payload)
            context = ToolContext(
                context=state, tool_name=tool.name,
                tool_call_id="invalid-assessment", tool_arguments=arguments,
            )
            result = await tool.on_invoke_tool(context, arguments)
            self.assertEqual(result["status"], "rejected")
            self.assertIsNone(state.lead_assessment)

    def test_issuer_coverage_is_checked_per_card_and_field(self):
        state = ResearchRun(hooks=None, researcher_agents={})
        cards = ["blue-cash-preferred", "blue-cash-everyday"]
        assignment = state.reserve_assignment(
            1, ISSUER, cards, "Compare rewards, annual fees, and current terms",
        )
        state.finish_assignment(assignment, succeeded=True)
        state.research_results.append({
            "agent": ISSUER,
            "status": "completed",
            "card_ids": cards,
            "task": assignment["task"],
            "tool_calls": [
                {
                    "name": "lookup_issuer_terms",
                    "args": {"card_id": "blue-cash-preferred"},
                    "result": {"evidence": [
                        {"field": "annual_fee_usd"},
                        {"field": "bonus_categories"},
                        {"field": "capped_categories"},
                    ]},
                },
                {
                    "name": "lookup_issuer_terms",
                    "args": {"card_id": "blue-cash-everyday"},
                    "result": {"evidence": [{"field": "annual_fee_usd"}]},
                },
            ],
            "unresearched_card_ids": [],
        })
        coverage = state.issuer_field_coverage()
        self.assertEqual(coverage["blue-cash-preferred"]["missing_fields"], [])
        self.assertEqual(
            coverage["blue-cash-everyday"]["missing_fields"],
            ["bonus_categories", "capped_categories"],
        )
        self.assertTrue(any(
            "blue-cash-everyday" in issue and "bonus_categories" in issue
            for issue in state.evidence_quality_issues()
        ))

    async def test_assessment_rejected_until_multicard_round_one_fanout_is_complete(self):
        state = ResearchRun(hooks=None, researcher_agents={})
        assignment = state.reserve_assignment(
            1,
            ISSUER,
            ["blue-cash-preferred", "blue-cash-everyday"],
            "Verify current terms",
        )
        state.finish_assignment(assignment, succeeded=True)
        lead = build_lead_agent("test-model")
        tool = next(tool for tool in lead.tools if tool.name == "save_research_assessment")
        payload = {
            "research_status": "incomplete",
            "verified_facts": [],
            "resolved_conflicts": [],
            "unresolved_gaps": ["Round-one research is incomplete."],
        }
        context = ToolContext(
            context=state,
            tool_name=tool.name,
            tool_call_id="incomplete-fanout",
            tool_arguments=json.dumps(payload),
        )
        await tool.on_invoke_tool(context, context.tool_arguments)
        self.assertIsNone(state.lead_assessment)

    async def test_completed_assessment_rejected_for_unresolved_calculation(self):
        state = ResearchRun(hooks=None, researcher_agents={})
        state.research_results.append({
            "agent": "Issuer Verification Researcher",
            "card_ids": ["blue-cash-preferred"],
            "task": "Calculate cash back for the user's spending.",
            "status": "completed",
            "unresearched_card_ids": [],
            "tool_calls": [{
                "name": "calculate_verified_rewards",
                "args": {"card_id": "blue-cash-preferred"},
                "result": {"status": "insufficient_evidence"},
            }],
        })
        lead = build_lead_agent("test-model")
        tool = next(tool for tool in lead.tools if tool.name == "save_research_assessment")
        payload = {
            "research_status": "completed",
            "verified_facts": [],
            "resolved_conflicts": [],
            "unresolved_gaps": [],
        }
        context = ToolContext(
            context=state,
            tool_name=tool.name,
            tool_call_id="reject-completed-assessment",
            tool_arguments=json.dumps(payload),
        )
        await tool.on_invoke_tool(context, context.tool_arguments)
        self.assertIsNone(state.lead_assessment)

        payload["research_status"] = "incomplete"
        context.tool_arguments = json.dumps(payload)
        result = await tool.on_invoke_tool(context, context.tool_arguments)
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(result["follow_up_required"])
        self.assertIsNone(state.lead_assessment)

        state.current_formal_round = MAX_RESEARCH_ROUNDS
        await tool.on_invoke_tool(context, context.tool_arguments)
        self.assertEqual(state.lead_assessment["research_status"], "incomplete")
        self.assertTrue(any(
            "calculation" in gap.casefold()
            for gap in state.lead_assessment["unresolved_gaps"]
        ))

    async def test_incomplete_assessment_stops_after_no_progress_follow_up(self):
        state = ResearchRun(hooks=None, researcher_agents={})
        failed_call = {
            "name": "calculate_from_terms",
            "args": {"card_id": "wf-active-cash", "spending": {"grocery": 500}},
            "result": {"card_id": "wf-active-cash", "status": "insufficient_evidence"},
        }
        state.research_results.extend([
            {
                "agent": ISSUER, "round": 1, "card_ids": ["wf-active-cash"],
                "task": "Calculate cash back", "status": "incomplete",
                "unresearched_card_ids": [], "tool_calls": [failed_call],
            },
            {
                "agent": ISSUER, "round": 2, "card_ids": ["wf-active-cash"],
                "task": "Repair missing calculation", "status": "incomplete",
                "unresearched_card_ids": [], "tool_calls": [failed_call],
            },
        ])
        state.current_formal_round = 2
        lead = build_lead_agent("test-model")
        tool = next(tool for tool in lead.tools if tool.name == "save_research_assessment")
        payload = {
            "research_status": "incomplete", "verified_facts": [],
            "resolved_conflicts": [], "unresolved_gaps": [],
        }
        arguments = json.dumps(payload)
        context = ToolContext(context=state, tool_name=tool.name,
                              tool_call_id="no-progress-stop", tool_arguments=arguments)
        await tool.on_invoke_tool(context, arguments)
        self.assertEqual(state.lead_assessment["stop_reason"], "no_progress")
        self.assertEqual(state.current_formal_round, 2)

    def test_later_estimated_calculation_resolves_earlier_failure(self):
        state = ResearchRun(hooks=None, researcher_agents={})
        base = {
            "agent": "Issuer Verification Researcher",
            "card_ids": ["blue-cash-preferred"],
            "task": "Calculate net rewards.",
            "status": "completed",
            "unresearched_card_ids": [],
        }
        state.research_results.extend([
            {**base, "tool_calls": [{
                "name": "calculate_verified_rewards",
                "args": {"card_id": "blue-cash-preferred"},
                "result": {"status": "insufficient_evidence"},
            }]},
            {**base, "tool_calls": [{
                "name": "calculate_verified_rewards",
                "args": {"card_id": "blue-cash-preferred"},
                "result": {"status": "estimated", "net_cash_back_usd": 301},
            }]},
        ])
        self.assertEqual(state.evidence_quality_issues(), [])

    async def test_open_market_discovery_is_delegated_to_rewards_researcher(self):
        researcher = object()
        state = ResearchRun(hooks=None, researcher_agents={REWARDS: researcher})
        lead = build_lead_agent("test-model")
        tool = next(tool for tool in lead.tools if tool.name == "discover_candidates")
        researcher_result = {
            "agent": REWARDS,
            "summary": "Two preliminary candidates.",
            "findings": [{"candidates": [{"card_id": "wf-active-cash"}]}],
            "tool_calls": [{
                "name": "screen_spending",
                "args": {"monthly_spend": {"dining": 500}},
                "result": {
                    "status": "requires_issuer_verification",
                    "candidates": [{
                        "card_id": "wf-active-cash",
                        "name": "Wells Fargo Active Cash",
                        "issuer": "Wells Fargo",
                        "estimated_net_rewards_usd": 120,
                        "unused_large_field": "x" * 10_000,
                    }],
                },
            }],
            "unresearched_card_ids": [],
        }
        context = ToolContext(
            context=state,
            tool_name=tool.name,
            tool_call_id="discovery-test",
            tool_arguments=json.dumps({"task": "Screen cards for $500 monthly dining spend."}),
        )
        with patch("scripts.lead.run_researcher", new=AsyncMock(return_value=researcher_result)) as run:
            await tool.on_invoke_tool(context, context.tool_arguments)
        run.assert_awaited_once_with(
            researcher,
            [],
            "Screen cards for $500 monthly dining spend.",
            hooks=None,
            trace=ANY,
        )
        self.assertEqual(state.research_results[0]["round"], 0)
        # Full evidence remains available to the Writer/shared state.
        self.assertIn("findings", state.research_results[0])
        self.assertIn("tool_calls", state.research_results[0])
        # The Lead sees a bounded handoff rather than duplicated raw evidence.
        lead_result = state.lead_tool_calls[-1]["result"]
        self.assertNotIn("findings", lead_result)
        self.assertNotIn("tool_calls", lead_result)
        candidate = lead_result["tool_outcomes"][0]["candidates"][0]
        self.assertEqual(candidate["card_id"], "wf-active-cash")
        self.assertNotIn("unused_large_field", candidate)

    async def test_clarification_skips_writer(self):
        lead_result = type("Result", (), {"final_output": "Please provide monthly spending."})()
        with patch("scripts.lead.Runner.run", new=AsyncMock(return_value=lead_result)), \
                patch("scripts.lead.generate_report", new=AsyncMock()) as writer:
            answer = await deep_research("Recommend a card")
        self.assertEqual(answer, "Please provide monthly spending.")
        writer.assert_not_awaited()

    async def test_writer_prompt_contains_lead_assessment(self):
        assessment = {
            "research_status": "incomplete",
            "verified_facts": ["Annual fee was verified."],
            "resolved_conflicts": [],
            "unresolved_gaps": ["Welcome bonus eligibility remains unknown."],
        }
        result = type("Result", (), {"final_output": "# Report"})()
        with patch("scripts.writer.Runner.run", new=AsyncMock(return_value=result)) as run:
            answer = await generate_report(
                "Compare cards",
                [{"agent": REWARDS, "findings": []}],
                assessment,
                model="test-model",
                model_settings=ModelSettings(),
            )
        writer_prompt = json.loads(run.await_args.args[1])
        self.assertEqual(answer, "# Report")
        self.assertEqual(writer_prompt["lead_assessment"], assessment)

    async def test_completed_research_is_passed_to_writer(self):
        finding = {"agent": "Rewards Researcher", "findings": [{"status": "catalog_snapshot"}]}
        assessment = {
            "research_status": "completed",
            "verified_facts": ["Rewards evidence was collected."],
            "resolved_conflicts": [],
            "unresolved_gaps": [],
        }

        async def run_lead(agent, prompt, *, context, **kwargs):
            context.research_results.append(finding)
            context.lead_assessment = assessment
            return type("Result", (), {"final_output": "RESEARCH_COMPLETE"})()

        with patch("scripts.lead.Runner.run", side_effect=run_lead), \
                patch("scripts.lead.generate_report", new=AsyncMock(return_value="# Report")) as writer:
            answer = await deep_research("Compare two cards")
        self.assertEqual(answer, "# Report")
        writer.assert_awaited_once_with(
            "Compare two cards",
            [finding],
            assessment,
            hooks=None,
            model=ANY,
            model_settings=ANY,
        )

    async def test_writer_is_blocked_when_assessment_is_missing(self):
        async def run_lead(agent, prompt, *, context, **kwargs):
            context.research_results.append({"agent": REWARDS, "findings": []})
            return type("Result", (), {"final_output": "RESEARCH_COMPLETE"})()

        with patch("scripts.lead.Runner.run", side_effect=run_lead), \
                patch("scripts.lead.generate_report", new=AsyncMock()) as writer:
            with self.assertRaisesRegex(RuntimeError, "without saving a Research Assessment"):
                await deep_research("Compare two cards")
        writer.assert_not_awaited()

    async def test_writer_is_blocked_without_completion_signal(self):
        async def run_lead(agent, prompt, *, context, **kwargs):
            context.research_results.append({"agent": REWARDS, "findings": []})
            context.lead_assessment = {
                "research_status": "completed",
                "verified_facts": [],
                "resolved_conflicts": [],
                "unresolved_gaps": [],
            }
            return type("Result", (), {"final_output": "DONE"})()

        with patch("scripts.lead.Runner.run", side_effect=run_lead), \
                patch("scripts.lead.generate_report", new=AsyncMock()) as writer:
            with self.assertRaisesRegex(RuntimeError, "required RESEARCH_COMPLETE signal"):
                await deep_research("Compare two cards")
        writer.assert_not_awaited()
