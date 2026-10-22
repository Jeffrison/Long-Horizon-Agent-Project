"""Credential-free integration fixture; this is not model evaluation."""
import json
from pathlib import Path
from agent.client import ScriptedClient
from agent.config import AgentConfig
from agent.react_loop import run_agent
from tools.types import ToolResult


def main():
    evidence = {"TestBook": "TestBook was written by Ada Example.",
                "Ada Example": "Ada Example was born in Exampleland."}
    client = ScriptedClient([
        "malformed fixture output",
        "Action: Search\nAction Input: TestBook",
        "Action: Search\nAction Input: Ada Example",
        "Final Answer: Exampleland"])
    result = run_agent("Which country is the author of TestBook from?",
                       config=AgentConfig(max_llm_calls=4), client=client,
                       tools={"Search": lambda query: ToolResult(True, evidence[query])})
    assert result["final_answer"] == "Exampleland"
    assert result["stats"]["llm_calls"] == 4
    assert result["stats"]["tool_calls"] == 2
    path = Path("outputs/demo/offline_fixture.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fixture_only": True, "result": result}, indent=2), encoding="utf-8")
    print(json.dumps({"fixture_only": True, "status": result["status"], "stats": result["stats"]}))


if __name__ == "__main__":
    main()
