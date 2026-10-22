"""Strict line-oriented ReAct parser; no heuristic answer extraction."""
import re


def check_gibberish(text):
    if not text or not text.strip():
        raise ValueError("Empty output")
    if re.search(r"(\b\w+\b)(?:\s+\1){19,}", text, flags=re.I):
        raise ValueError("Repetitive words")
    if re.search(r"(\S.{0,20}?)\1{19,}", text, flags=re.S):
        raise ValueError("Repetitive pattern")


def parse_llm_output(text, task_type="QA"):
    text = text.strip()
    finals = re.findall(r"^Final\s+Answer:\s*(.+)$", text, flags=re.I | re.M)
    actions = re.findall(r"^Action:\s*([^\n]+)", text, flags=re.I | re.M)
    if finals:
        if len(finals) != 1 or actions:
            raise ValueError("Return exactly one final answer or one action")
        answer = finals[0].strip()
        check_gibberish(answer)
        return {"type": "finish", "content": answer}
    inputs = re.search(r"^Action\s+Input:\s*(.*)", text, flags=re.I | re.M | re.S)
    if len(actions) != 1 or inputs is None:
        raise ValueError("Missing Action and Action Input")
    action = actions[0].strip().strip(chr(96) + "*\"' ")
    value = inputs.group(1).strip()
    fence = chr(96) * 3
    if value.startswith(fence):
        value = "\n".join(value.splitlines()[1:]).removesuffix(fence).strip()
    check_gibberish(value)
    return {"type": "action", "action": action, "action_input": value}
