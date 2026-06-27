# tests/test_io_utils.py
import json
from pathlib import Path
from src import io_utils

SAMPLE = Path(__file__).parent / "fixtures_sample_candidates.json"

def _first():
    return json.load(open(SAMPLE, encoding="utf-8"))[0]

def test_iter_candidates_streams_jsonl(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text('{"candidate_id":"CAND_0000001"}\n\n{"candidate_id":"CAND_0000002"}\n', encoding="utf-8")
    ids = [c["candidate_id"] for c in io_utils.iter_candidates(f)]
    assert ids == ["CAND_0000001", "CAND_0000002"]   # blank line skipped

def test_build_profile_text_includes_key_fields():
    c = _first()
    text = io_utils.build_profile_text(c)
    assert c["profile"]["current_title"].lower() in text.lower()
    assert len(text) > 50
    # includes at least one skill name
    assert c["skills"][0]["name"].lower() in text.lower()
