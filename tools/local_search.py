"""Deterministic BM25 over a question's ten frozen, unlabeled candidate paragraphs."""
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from tools.result import ToolResult


def tokenize(text):
    return re.findall(r"\w+", text.casefold())


@lru_cache(maxsize=4)
def load_contexts(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(set(row) != {"question_key", "documents"} for row in rows):
        raise ValueError("Context files may contain only question keys and documents")
    if any(set(doc) != {"title", "text"} for row in rows for doc in row["documents"]):
        raise ValueError("Document fields must be title/text")
    return {row["question_key"]: row["documents"] for row in rows}


class LocalSearch:
    def __init__(self, documents):
        self.documents = documents
        self.counts = [Counter(tokenize(d["title"] + " " + d["title"] + " " + d["text"])) for d in documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average_length = sum(self.lengths) / max(1, len(documents))
        self.document_frequency = Counter(word for count in self.counts for word in count)

    def __call__(self, query, max_results=2):
        tokens = set(tokenize(query))
        if not tokens:
            return ToolResult("Empty search query", True, "invalid_input")
        scored = []
        total = len(self.documents)
        for index, (doc, count, length) in enumerate(zip(self.documents, self.counts, self.lengths)):
            score = 0.0
            for token in tokens:
                frequency = count[token]
                idf = math.log(1 + (total-self.document_frequency[token]+0.5)/(self.document_frequency[token]+0.5))
                score += idf * frequency * 2.5 / (frequency+1.5*(0.25+0.75*length/max(1,self.average_length)))
            if tokenize(query) == tokenize(doc["title"]):
                score += 10
            if score > 0:
                scored.append((score, doc["title"], index, doc))
        scored.sort(key=lambda row: (-row[0], row[1], row[2]))
        selected = scored[:max_results]
        if not selected:
            return ToolResult("No matching paragraphs in this question's frozen candidate corpus.", cached=True)
        return ToolResult("\n\n".join(f"Title: {doc['title']}\nSnippet: {doc['text'][:2800]}"
                                      for _, _, _, doc in selected), cached=True)
