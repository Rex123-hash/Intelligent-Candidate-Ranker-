"""Structured JD-fit feature scores, each in [0, 1]. Derived directly from the JD."""
from typing import List, Dict, Any
from src import config as C

def experience_score(yoe: float) -> float:
    """Triangular peak at EXP_IDEAL, linear falloff, floored at 0.1."""
    s = 1.0 - abs(yoe - C.EXP_IDEAL) / C.EXP_HALF_WIDTH
    return max(0.1, min(1.0, s)) if s > 0 else 0.1

def product_company_score(career_history: List[Dict[str, Any]]) -> float:
    """Share of career months spent at non-services (product) companies."""
    total = 0.0
    product = 0.0
    for role in career_history:
        m = float(role.get("duration_months", 0) or 0)
        total += m
        name = (role.get("company", "") or "").lower()
        if not any(f in name for f in C.SERVICES_FIRMS):
            product += m
    if total == 0:
        return 0.5
    return product / total

def title_score(current_title: str, career_history: List[Dict[str, Any]]) -> float:
    """Best positive title-keyword match minus negative-keyword presence."""
    titles = [current_title or ""]
    titles += [r.get("title", "") for r in career_history[:3]]
    blob = " ".join(t.lower() for t in titles)
    pos = 0.0
    for kw, w in C.RELEVANT_TITLE_KEYWORDS.items():
        if kw in blob:
            pos = max(pos, w)
    neg = any(kw in (current_title or "").lower() for kw in C.NEGATIVE_TITLE_KEYWORDS)
    if pos == 0.0 and neg:
        return 0.1
    if neg:
        pos *= 0.5
    return max(0.1, min(1.0, pos)) if pos else 0.25

def location_score(location: str, willing_to_relocate: bool) -> float:
    """Reward target Indian hubs; partial for other India; relocation lifts others."""
    loc = (location or "").lower()
    if any(city in loc for city in C.TARGET_CITIES):
        return 1.0
    india_states = ("india", "maharashtra", "karnataka", "telangana", "tamil",
                    "kerala", "haryana", "uttar pradesh", "delhi", "gujarat",
                    "punjab", "odisha", "west bengal", "rajasthan", "chandigarh")
    in_india = any(s in loc for s in india_states)
    base = 0.6 if in_india else 0.2
    if willing_to_relocate:
        base = min(1.0, base + 0.3)
    return base

def education_score(education: List[Dict[str, Any]]) -> float:
    """Minor signal from best institution tier."""
    tier_map = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.3,
                "unknown": 0.5}
    if not education:
        return 0.5
    return max(tier_map.get(e.get("tier", "unknown"), 0.5) for e in education)

def feature_subscore(c: Dict[str, Any]) -> Dict[str, float]:
    """Return all feature components plus their weighted aggregate in [0,1]."""
    p = c.get("profile", {})
    ch = c.get("career_history", [])
    sig = c.get("redrob_signals", {})
    comps = {
        "experience": experience_score(float(p.get("years_of_experience", 0) or 0)),
        "product": product_company_score(ch),
        "title": title_score(p.get("current_title", ""), ch),
        "location": location_score(p.get("location", ""), bool(sig.get("willing_to_relocate", False))),
        "education": education_score(c.get("education", [])),
    }
    agg = (C.F_EXPERIENCE * comps["experience"] + C.F_PRODUCT * comps["product"]
           + C.F_TITLE * comps["title"] + C.F_LOCATION * comps["location"]
           + C.F_EDUCATION * comps["education"])
    comps["aggregate"] = agg
    return comps
