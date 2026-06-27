"""Red-flag penalties from the JD's explicit 'do NOT want' list.
Returns (total_penalty_in_[0,1], list_of_reason_strings)."""
from typing import Dict, Any, List, Tuple
from src import config as C

# individual penalty magnitudes (subtractive, before W_PENALTY scaling)
P_CONSULTING = 0.40
P_MANAGER_NOCODE = 0.30
P_CV_SPEECH = 0.30
P_LANGCHAIN_ONLY = 0.25
P_TITLE_CHASE = 0.20

def _all_services(ch: List[Dict[str, Any]]) -> bool:
    if not ch:
        return False
    return all(any(f in (r.get("company", "") or "").lower() for f in C.SERVICES_FIRMS)
               for r in ch)

def _skill_names(c) -> str:
    return " ".join((s.get("name", "") or "").lower() for s in c.get("skills", []))

def penalties(c: Dict[str, Any]) -> Tuple[float, List[str]]:
    p = c.get("profile", {})
    ch = c.get("career_history", [])
    title = (p.get("current_title", "") or "").lower()
    skills_blob = _skill_names(c)
    total = 0.0
    reasons: List[str] = []

    # 1. Entirely consulting/services career
    if _all_services(ch):
        total += P_CONSULTING
        reasons.append("Career entirely at consulting/services firms")

    # 2. Manager/architect with no recent hands-on engineering title.
    # A recent role counts as hands-on only if it has an IC-engineer keyword AND is
    # not itself a management title (otherwise "Engineering Manager" / "Director of
    # Engineering" falsely match the "engineer" substring inside "engineering").
    is_manager = any(k in title for k in C.MANAGER_TITLE_KEYWORDS)

    def _is_ic_engineer(role_title: str) -> bool:
        t = (role_title or "").lower()
        return (any(k in t for k in C.ENGINEER_TITLE_KEYWORDS)
                and not any(m in t for m in C.MANAGER_TITLE_KEYWORDS))

    codes_recently = any(_is_ic_engineer(r.get("title", "")) for r in ch[:2])
    if is_manager and not codes_recently:
        total += P_MANAGER_NOCODE
        reasons.append("Senior management title with no recent hands-on engineering")

    # 3. CV/speech/robotics focus with no NLP/IR exposure
    has_cv = any(k in skills_blob for k in C.CV_SPEECH_KEYWORDS)
    has_nlp = any(k in skills_blob for k in C.AI_SKILL_KEYWORDS)
    if has_cv and not has_nlp:
        total += P_CV_SPEECH
        reasons.append("Vision/speech/robotics focus without NLP/IR")

    # 4. LLM-wrapper-only: LangChain present, low tenure, no deeper ML history
    has_langchain = any(k in skills_blob for k in C.LANGCHAIN_KEYWORDS)
    deep_ml_months = max([float(s.get("duration_months", 0) or 0)
                          for s in c.get("skills", [])
                          if any(k in (s.get("name", "") or "").lower()
                                 for k in C.AI_SKILL_KEYWORDS)] + [0])
    if has_langchain and deep_ml_months < 12:
        total += P_LANGCHAIN_ONLY
        reasons.append("AI experience appears limited to recent LLM-wrapper work")

    # 5. Title-chasing: 3+ short (<18 mo) stints
    short = sum(1 for r in ch if 0 < float(r.get("duration_months", 0) or 0) < 18)
    if short >= 3:
        total += P_TITLE_CHASE
        reasons.append("Frequent short job stints (possible title-chasing)")

    return min(total, 1.0), reasons
