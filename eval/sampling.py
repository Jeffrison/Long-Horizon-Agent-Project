"""Sampling summaries require all n attempts, not just a retained successful record."""
from collections import Counter
from math import comb
from eval.metrics import normalize_text


def pass_at_k(n, correct, k):
    # HumanEval estimator; meaningful only for a complete, fixed sampling protocol.
    if not (isinstance(n, int) and isinstance(correct, int) and isinstance(k, int)
            and n > 0 and 0 <= correct <= n and 1 <= k <= n):
        raise ValueError("Need complete n, 0<=correct<=n, and 1<=k<=n")
    return 1.0 - comb(n - correct, k) / comb(n, k) if n - correct >= k else 1.0


def select_without_reference(attempts, policy="majority"):
    """Returns an index using predictions only; never inspects correctness/scores."""
    if policy not in {"first", "majority"}:
        raise ValueError("Invalid selector")
    if policy == "first":
        return 0 if attempts else None
    answers = [(index, normalize_text(item["result"].get("final_answer")))
               for index, item in enumerate(attempts)
               if item["result"]["status"] == "success"]
    answers = [(index, answer) for index, answer in answers if answer]
    if not answers:
        return None
    counts = Counter(answer for _, answer in answers)
    winning = max(counts.values())
    # Deterministic tie break: first candidate among tied answers.
    return next(index for index, answer in answers if counts[answer] == winning)
