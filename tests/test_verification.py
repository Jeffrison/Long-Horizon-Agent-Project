from dataclasses import asdict
import json
import pytest
from agent.react_loop import AgentConfig, run_agent
from eval.data_loader import question_key, split_for
from eval.verify_run import verify
from tests.test_agent import ScriptedClient


def test_frozen_manifest_rejects_label_tampering(tmp_path):
    query = "Compute 6 * 7."
    item = {"id":"q","question":query,"answer":"42","question_key":question_key(query),"split":split_for(query)}
    config = AgentConfig(seed=11,max_calls=1)
    result = run_agent(query,"MATH",config=config,enable_reflection=False,
                       client=ScriptedClient(["Final Answer: 42"]),tools={})
    manifest = {"settings":{"tasks":{"MATH":[item]},"seeds":[11],"modes":["off"],"config":asdict(config)}}
    row = {"key":"MATH:q:11:off","task":"MATH","id":"q","seed":11,"mode":"off",
           "question":query,"question_key":item["question_key"],"split":item["split"],
           "ground_truth":"42","is_correct":True,"result":result}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    path=tmp_path/"episodes.jsonl"
    path.write_text(json.dumps(row)+"\n")
    assert verify(tmp_path)["complete"]
    row["ground_truth"]="43"
    path.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="labels/identity"):
        verify(tmp_path)
