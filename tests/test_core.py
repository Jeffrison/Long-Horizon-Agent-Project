import ast
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from agent.client import ScriptedClient
from agent.config import AgentConfig
from agent.parser import parse_llm_output
from agent.react_loop import run_agent
from eval.data_loader import load_jsonl_dataset, split_for_question
from eval.metrics import math_match, parse_latex_to_float, qa_match, qa_scores
from eval.run_eval import evaluate
from tools.python_repl import auto_inject_print, run_python
from tools.types import ToolResult


class MetricsTests(unittest.TestCase):
    def test_no_substring_or_yes_no_shortcut(self):
        self.assertFalse(qa_match("not sure; both countries are different", "yes"))
        self.assertFalse(qa_match("It might be Charmed or another show", "Charmed"))
        self.assertEqual(qa_scores("Yes, definitely", "yes")["f1"], 0)

    def test_hotpot_normalization_and_token_f1(self):
        self.assertTrue(qa_match("The U.S.", "US"))
        self.assertEqual(qa_scores("red blue", "red green")["f1"], 0.5)

    def test_math_is_scalar_not_traceback_number(self):
        self.assertTrue(math_match("0.25", r"\frac{1}{4}"))
        self.assertTrue(math_match("0", 0))
        self.assertFalse(math_match("Error at line 70", 70))
        self.assertIsNone(parse_latex_to_float("sqrt(-1)"))
        self.assertIsNone(parse_latex_to_float("2**100000"))
        self.assertIsNone(parse_latex_to_float("__import__('os').getcwd()"))


class AgentTests(unittest.TestCase):
    def test_retry_consumes_call_budget_and_invalid_text_is_not_remembered(self):
        client = ScriptedClient(["BROKEN OUTPUT", "still broken"])
        result = run_agent("q", config=AgentConfig(max_llm_calls=2), client=client,
                           tools={"Search": lambda q: ToolResult(True, "evidence")})
        self.assertEqual(result["stats"]["llm_calls"], 2)
        self.assertEqual(result["termination_reason"], "llm_budget")
        self.assertNotIn("BROKEN OUTPUT", json.dumps(client.requests[1]))
        self.assertIsNone(result["stats"]["prompt_tokens"])

    def test_tool_exception_is_visible_and_recoverable(self):
        client = ScriptedClient(["Action: search\nAction Input: q", "Final Answer: answer"])
        result = run_agent("q", client=client,
                           tools={"Search": lambda q: ToolResult(False, "Try again", "network_error")})
        self.assertEqual(result["final_answer"], "answer")
        self.assertEqual(result["stats"]["errors"]["network_error"], 1)
        self.assertEqual(result["stats"]["tool_calls"], 1)

    def test_error_word_in_evidence_is_not_a_tool_failure(self):
        client = ScriptedClient(["Action: Search\nAction Input: Error", "Final Answer: book"])
        result = run_agent("q", config=AgentConfig(mode="react"), client=client,
                           tools={"Search": lambda q: ToolResult(True, "Error is the book title")})
        self.assertEqual(result["status"], "success")

    def test_unknown_tool_and_tool_budget(self):
        client = ScriptedClient(["Action: fake\nAction Input: q"] * 3)
        result = run_agent("q", config=AgentConfig(max_tool_calls=1), client=client,
                           tools={"Search": lambda q: ToolResult(True, "x")})
        self.assertEqual(result["termination_reason"], "tool_budget")
        self.assertEqual(result["stats"]["tool_calls"], 1)

    def test_no_gold_parameter_or_inference_leak(self):
        self.assertNotIn("ground_truth", inspect.signature(run_agent).parameters)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text(json.dumps({"question": "Find an entity", "answer": "SECRET_GOLD_VALUE"}) + "\n")
            client = ScriptedClient(["Final Answer: wrong"])
            report = evaluate(path, "QA", AgentConfig(mode="direct"), split="all", client=client)
        self.assertNotIn("SECRET_GOLD_VALUE", json.dumps(client.requests))
        self.assertEqual(report["metrics"]["em"], 0)

    def test_api_failure_is_counted_without_exposing_exception_text(self):
        result = run_agent("q", config=AgentConfig(mode="direct"),
                           client=ScriptedClient([RuntimeError("secret credential")]))
        self.assertEqual(result["stats"]["llm_calls"], 1)
        self.assertNotIn("secret credential", json.dumps(result))

    def test_direct_mode_cannot_call_tools(self):
        result = run_agent("q", config=AgentConfig(mode="direct"),
                           client=ScriptedClient(["Action: Search\nAction Input: q"]))
        self.assertEqual(result["termination_reason"], "format_error")

    def test_conflicting_final_answers_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_llm_output("Final Answer: yes\nFinal Answer: no")


class DataAndExecutionTests(unittest.TestCase):
    def test_zero_answers_and_repeated_source_ids_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('\n'.join(json.dumps(x) for x in [
                {"id": "0", "question": "q1", "answer": 0},
                {"id": "0", "question": "q2", "answer": 1}]))
            rows = load_jsonl_dataset(path)
        self.assertEqual(rows[0]["answer"], "0")
        self.assertNotEqual(rows[0]["id"], rows[1]["id"])
        self.assertEqual(split_for_question(" Same QUESTION "), split_for_question("same question"))

    def test_ast_expression_is_executed_once(self):
        code = "items = []\nitems.append(1)"
        tree = ast.parse(auto_inject_print(code))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertEqual(len(calls), 1)
        result = run_python("values=[]\ndef f():\n values.append(1)\n return len(values)\nf()")
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "1")

    def test_local_execution_rejects_io_and_observes_timeout(self):
        self.assertEqual(run_python("import os\nos.getcwd()").error, "invalid_code")
        self.assertEqual(run_python("while True: pass", timeout=0.1).error, "timeout")


if __name__ == "__main__":
    unittest.main()
