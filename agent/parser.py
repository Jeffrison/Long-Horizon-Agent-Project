"""Strict protocol parsing without answer-specific validation or code rewriting."""
import re


def check_gibberish(text):
    if re.search(r"(\S.{0,20}?)\1{19,}", text, re.DOTALL):
        raise ValueError("Repetitive output")


def parse_llm_output(text: str, task_type: str = "QA") -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty output")
    check_gibberish(text)
    outside, fenced = [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            outside.append("")
        else:
            outside.append("" if fenced else line)
    protocol = "\n".join(outside)
    finals = list(re.finditer(r"^[ \t]*Final\s+Answer:[ \t]*([^\n]*)", protocol, re.I | re.M))
    actions = list(re.finditer(r"^[ \t]*Action:[ \t]*([^\n]+)", protocol, re.I | re.M))
    if len(finals) == 1 and not actions:
        answer = finals[0].group(1).strip()
        if not answer:
            raise ValueError("Empty final answer")
        return {"type": "finish", "content": answer}
    if len(actions) != 1 or finals:
        raise ValueError("Expected exactly one action or final answer")
    inputs = list(re.finditer(r"^[ \t]*Action Input:[ \t]*(.*)", text, re.I | re.M))
    if len(inputs) != 1:
        raise ValueError("Missing or repeated Action Input")
    action_input = text[inputs[0].start():].split(":", 1)[1].strip()
    if action_input.startswith("```"):
        match = re.fullmatch(r"```(?:python)?[ \t]*\n(.*?)\n```", action_input, re.I | re.S)
        if not match:
            raise ValueError("Unclosed or invalid code fence")
        action_input = match.group(1)
    if not action_input.strip():
        raise ValueError("Empty action input")
    return {"type": "action", "action": actions[0].group(1).strip(), "action_input": action_input}
