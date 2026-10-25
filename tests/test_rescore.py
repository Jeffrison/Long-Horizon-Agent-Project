from dataclasses import asdict, replace
import json
from agent.react_loop import AgentConfig, run_agent
from eval.data_loader import question_key, split_for
from eval.rescore import rescore
from tests.test_agent import ScriptedClient


def test_rescoring_changes_only_grade_not_saved_prediction(tmp_path):
    source=tmp_path/"original"
    source.mkdir()
    query="Compute the count."
    item={"id":"q","question":query,"answer":"1000000","question_key":question_key(query),"split":split_for(query)}
    config=AgentConfig(max_calls=1)
    result=run_agent(query,"MATH",config=replace(config,seed=11),enable_reflection=False,
                     client=ScriptedClient(["Final Answer: 1000001"]),tools={})
    settings={"tasks":{"MATH":[item]},"seeds":[11],"modes":["off"],"config":asdict(config),
              "metric_version":"strict-v2","split":"all","protocol":"unit-test"}
    manifest={"settings":settings,"git_revision":"original-test-source"}
    row={"key":"MATH:q:11:off","task":"MATH","id":"q","seed":11,"mode":"off", "question":query,
         "question_key":item["question_key"],"split":item["split"],"ground_truth":"1000000",
         "is_correct":True,"answer_f1":None,"result":result}
    (source/"manifest.json").write_text(json.dumps(manifest))
    (source/"episodes.jsonl").write_text(json.dumps(row)+"\n")
    (source/"started.jsonl").write_text(json.dumps({"key":row["key"]})+"\n")
    original=(source/"episodes.jsonl").read_bytes()
    changes=rescore(source,tmp_path/"rescored")
    assert len(changes)==1
    assert (source/"episodes.jsonl").read_bytes()==original
    new=json.loads((tmp_path/"rescored/episodes.jsonl").read_text())
    assert new["result"]==result and not new["is_correct"] and new["original_is_correct"]
    meta=json.loads((tmp_path/"rescored/manifest.json").read_text())
    assert meta["rescoring"]["model_calls_made_by_rescoring"]==0
