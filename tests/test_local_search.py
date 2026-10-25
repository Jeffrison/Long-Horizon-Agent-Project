import json
import pytest
from tools.local_search import LocalSearch, load_contexts


def test_local_bm25_returns_frozen_evidence_deterministically():
    tool = LocalSearch([{"title":"David Yates","text":"David Yates is an English director."},
                        {"title":"David Bowie","text":"David Bowie is a musician."}])
    first = tool("David Yates")
    assert first.content.startswith("Title: David Yates")
    assert tool("David Yates") == first
    assert first.cached and not first.is_error
    assert tool("").is_error


def test_tool_visible_context_rejects_label_fields(tmp_path):
    path = tmp_path/"contexts.jsonl"
    path.write_text(json.dumps({"question_key":"q","documents":[],"answer":"secret"})+"\n")
    with pytest.raises(ValueError,match="only"):
        load_contexts(str(path))
