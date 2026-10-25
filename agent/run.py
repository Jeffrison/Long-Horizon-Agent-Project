"""Run one API or local-model Agent task, independently of scoring."""
import argparse
import json
from pathlib import Path
from agent.react_loop import AgentConfig, run_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--task", choices=["QA","MATH"], default="MATH")
    parser.add_argument("--backend", choices=["api","local"], default="api")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--adapter")
    parser.add_argument("--reflection", choices=["off","on"], default="on")
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        raise ValueError("Refusing to overwrite a previous trajectory")
    config = AgentConfig(model=args.model,max_calls=args.max_calls,max_tokens=args.max_tokens,seed=args.seed)
    client = None
    if args.backend == "local":
        from agent.local_client import LocalChatClient
        client = LocalChatClient(config,args.revision,args.adapter)
    result = run_agent(args.query, args.task, config=config,client=client,
                       enable_reflection=args.reflection == "on")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in {"events","trajectory","initial_messages","config"}},
                     ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
