"""Edge-case / robustness tests: empty or partial candidate records must never
crash the scoring pipeline, and every sub-score must stay within its declared
range. These guard against malformed rows in a 100k real-world pool."""
from src import features as F
from src import flags as FL
from src import signals as SG
from src import honeypot as HP
from src import scoring
from src import reasoning
from src import io_utils
from src import config as C


# --- features stay in range on empty / missing inputs ---

def test_feature_subscore_on_empty_candidate_in_range():
    comps = F.feature_subscore({})
    for k, v in comps.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of range"


def test_experience_score_bounds_extremes():
    for yoe in (0, 1, 7, 13, 30, 50):
        assert 0.0 <= F.experience_score(float(yoe)) <= 1.0


def test_product_score_empty_history_is_neutral():
    assert F.product_company_score([]) == 0.5


def test_title_score_empty_is_neutral_low():
    assert 0.0 < F.title_score("", []) < 0.5


def test_location_empty_is_low_but_valid():
    assert 0.0 <= F.location_score("", False) <= 1.0


def test_relocation_never_exceeds_one():
    assert F.location_score("Pune", True) == 1.0  # already max, no overflow


# --- penalties / behavior / honeypot robustness ---

def test_penalties_on_empty_candidate():
    pen, reasons = FL.penalties({"profile": {}, "career_history": [], "skills": []})
    assert pen == 0.0 and reasons == []


def test_behavior_modifier_missing_signals_in_range():
    m = SG.behavior_modifier({})
    assert C.BEHAVIOR_FLOOR <= m <= 1.0


def test_honeypot_empty_is_false():
    assert HP.is_honeypot({}) is False


def test_honeypot_handles_unparseable_dates():
    # Consistent tenure (60 mo == 5 yrs) so only the date-parse path is exercised:
    # an unparseable start date must NOT trip the elapsed-tenure check (no false positive).
    c = {"career_history": [{"start_date": "not-a-date", "end_date": None,
                             "duration_months": 60, "is_current": True}],
         "skills": [], "profile": {"years_of_experience": 5}}
    assert HP.is_honeypot(c) is False


# --- end-to-end scoring never crashes and stays bounded ---

def test_score_candidate_minimal_record():
    c = {"candidate_id": "CAND_0000001", "profile": {}, "career_history": [],
         "skills": [], "education": [], "redrob_signals": {}}
    s = scoring.score_candidate(c, semantic=0.5)
    assert 0.0 <= s["score"] <= 1.0
    assert set(["candidate_id", "score", "is_honeypot", "components", "flag_reasons"]) <= set(s)


def test_score_candidate_negative_semantic_bounded():
    c = {"candidate_id": "CAND_0000002", "profile": {"current_title": "ML Engineer",
         "years_of_experience": 7}, "career_history": [], "skills": [],
         "education": [], "redrob_signals": {}}
    s = scoring.score_candidate(c, semantic=-0.4)
    assert 0.0 <= s["score"] <= 1.0


def test_build_profile_text_minimal_no_crash():
    text = io_utils.build_profile_text({"profile": {}, "career_history": [], "skills": []})
    assert isinstance(text, str)


def test_build_reason_minimal_no_crash():
    c = {"candidate_id": "CAND_0000003", "profile": {}, "career_history": [],
         "skills": [], "redrob_signals": {}}
    s = scoring.score_candidate(c, semantic=0.5)
    r = reasoning.build_reason(c, s, rank=50, variant=2)
    assert isinstance(r, str) and len(r) <= 240
