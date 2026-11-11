import unittest

from context_panel import AgentContexts, PanelHooks


class ActivityEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifecycle_and_research_actions_are_distinct(self):
        events = []
        hooks = PanelHooks(AgentContexts(), events.append)
        agent = type("Agent", (), {"name": "Rewards Researcher"})()
        research_tool = type("Tool", (), {"name": "lookup_rewards"})()
        orchestration_tool = type("Tool", (), {"name": "delegate_rewards"})()

        await hooks.on_agent_start(None, agent)
        await hooks.on_tool_start(None, agent, research_tool)
        await hooks.on_tool_start(None, agent, orchestration_tool)
        await hooks.on_agent_end(None, agent, "done")

        self.assertEqual(
            [event["kind"] for event in events],
            ["agent_lifecycle", "research_action", "orchestration", "agent_lifecycle"],
        )
        self.assertEqual(events[1]["action"], "lookup_rewards")
        self.assertEqual(events[2]["action"], "delegate_rewards")


if __name__ == "__main__":
    unittest.main()
