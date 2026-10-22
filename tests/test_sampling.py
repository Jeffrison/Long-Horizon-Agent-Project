import inspect
import unittest

from agent.client import ScriptedClient
from agent.config import AgentConfig
from agent.react_loop import run_agent
from eval.fair_compare import generate_question, score_question
from eval.provenance import source_policy
from eval.sampling import pass_at_k
from tools.types import ToolResult


class SamplingTests(unittest.TestCase):
    def test_complete_sampling_pass_at_k(self):
        self.assertAlmostEqual(pass_at_k(5, 2, 1), 0.4)
        self.assertAlmostEqual(pass_at_k(5, 2, 2), 0.7)
        self.assertEqual(pass_at_k(5, 2, 5), 1)
        self.assertEqual(pass_at_k(5, 0, 5), 0)
        with self.assertRaises(ValueError):
            pass_at_k(1, 1, 8)

    def test_all_attempts_retained_after_first_answer_is_correct(self):
        seen = []
        def factory(strategy, index):
            answer = "Paris" if index == 0 else "London"
            return ScriptedClient(["Final Answer: " + answer])
        generated = generate_question("capital?", AgentConfig(max_tool_calls=6), factory,
                                      lambda: {"Search": lambda q: ToolResult(True, "evidence")},
                                      total_calls=6, restarts=3, sink=seen.append)
        self.assertEqual(len(seen), 5)  # single + feedback + all 3 restarts
        self.assertEqual(len(generated["react_restarts"]), 3)
        scored = score_question(generated, "Paris")["react_restarts"]
        self.assertTrue(scored["oracle_any_success"])
        self.assertEqual(scored["selected_prediction"], "London")
        self.assertEqual(scored["selected_scores"]["em"], 0)
        self.assertFalse(scored["oracle_used_for_selection"])
        self.assertNotIn("reference", inspect.signature(generate_question).parameters)

    def test_retry_budget_and_restart_allocation(self):
        generated = generate_question("q", AgentConfig(max_tool_calls=8),
                                      lambda strategy, index: ScriptedClient(["malformed"] * 10),
                                      lambda: {"Search": lambda q: ToolResult(True, "evidence")},
                                      total_calls=8, restarts=4)
        for attempts in generated.values():
            self.assertLessEqual(sum(x["result"]["stats"]["llm_calls"] for x in attempts), 8)
        self.assertEqual([x["allocated_llm_calls"] for x in generated["react_restarts"]], [2] * 4)
        with self.assertRaises(ValueError):
            generate_question("q", AgentConfig(), None, None, total_calls=7, restarts=4)

    def test_seeds_are_explicit_per_trajectory_not_client_global_state(self):
        class Recorder:
            def __init__(self):
                self.seeds = []
            def complete(self, messages, config):
                self.seeds.append(config.seed)
                return {"text": "malformed", "usage": {}}
        client = Recorder()
        for _ in range(2):
            result = run_agent("q", config=AgentConfig(max_llm_calls=3, seed=123), client=client,
                               tools={"Search": lambda q: ToolResult(True, "evidence")})
            self.assertEqual([e["effective_seed"] for e in result["events"]], [123, 124, 125])
        self.assertEqual(client.seeds, [123, 124, 125, 123, 124, 125])

    def test_known_selected_source_cannot_be_treated_as_unbiased(self):
        policy = source_policy("560074abd1fc686a5c524ee4086b2b89a654ab511b716287f21ae67d404b9440")
        self.assertTrue(policy["selection_conditioned_training_data"])
        self.assertFalse(policy["all_attempts_recorded"])
        self.assertFalse(policy["eligible_for_reflection_gain_comparison"])


if __name__ == "__main__":
    unittest.main()
