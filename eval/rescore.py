"""Regrade immutable saved predictions; never calls a model or chooses a new answer."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from eval.run_eval import read_rows
from eval.scoring import score_answer,VERSIONS
from eval.verify_run import verify


def rescore(source,destination,version="strict-v3"):
    source,destination=Path(source),Path(destination)
    verified=verify(source)
    if not verified["complete"]:
        raise ValueError("Only complete planned runs can be rescored")
    manifest=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    rows=read_rows(source/"episodes.jsonl")
    new_manifest=copy.deepcopy(manifest)
    original_version=manifest["settings"]["metric_version"]
    new_manifest["settings"]["metric_version"]=version
    new_manifest["rescoring"]={
        "source_manifest_sha256":hashlib.sha256((source/"manifest.json").read_bytes()).hexdigest(),
        "source_episodes_sha256":hashlib.sha256((source/"episodes.jsonl").read_bytes()).hexdigest(),
        "original_metric_version":original_version,"new_metric_version":version,
        "scoring_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "scorer_sha256":hashlib.sha256(Path("eval/exact_math.py").read_bytes()).hexdigest(),
        "model_calls_made_by_rescoring":0}
    differences=[]
    for row in rows:
        old=row["is_correct"]
        new=bool(row["result"]["status"]=="success" and
                 score_answer(row["result"].get("final_answer"),row["ground_truth"],row["task"],version))
        row["original_is_correct"]=old
        row["is_correct"]=new
        if old!=new:
            differences.append({"key":row["key"],"prediction":row["result"].get("final_answer"),
                                "ground_truth":row["ground_truth"],"old_correct":old,"new_correct":new})
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"manifest.json").write_text(json.dumps(new_manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    (destination/"episodes.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf-8")
    shutil.copyfile(source/"started.jsonl",destination/"started.jsonl")
    (destination/"scoring-differences.json").write_text(json.dumps(differences,ensure_ascii=False,indent=2),encoding="utf-8")
    from eval.report import write_report
    write_report(destination)
    verify(destination)
    print(json.dumps({"rescored":len(rows),"changed":len(differences),"new_version":version,"model_calls":0},indent=2))
    return differences


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output",required=True)
    parser.add_argument("--version",choices=VERSIONS,default="strict-v3")
    args=parser.parse_args()
    rescore(args.source,args.output,args.version)
