"""Build a 1-2 sentence, fact-grounded reason for each ranked candidate.
Every clause is derived from fields present in the candidate's profile or the
score components -- no invented skills or employers."""
from typing import Dict, Any, List
from src import config as C

_OPENERS = [
    "{yoe} yrs as {title}",
    "{title} with {yoe} yrs experience",
    "{yoe}-year {title}",
]

def _ai_skills_present(c) -> List[str]:
    out = []
    for s in c.get("skills", []):
        name = (s.get("name", "") or "")
        if any(k in name.lower() for k in C.AI_SKILL_KEYWORDS):
            out.append(name)
    return out[:3]

def _concern(c: Dict[str, Any], sc: Dict[str, Any]) -> str:
    sig = c.get("redrob_signals", {})
    if sc["flag_reasons"]:
        return sc["flag_reasons"][0].lower()
    npd = sig.get("notice_period_days")
    if npd is not None and float(npd) >= 90:
        return f"long notice period ({int(npd)} days)"
    if float(sig.get("recruiter_response_rate", 1.0) or 0) < 0.3:
        return "low recruiter response rate"
    if sc["components"].get("location", 1.0) < 0.5:
        return "location/relocation may be a constraint"
    if sc["components"].get("product", 1.0) < 0.4:
        return "mostly services-company background"
    return ""

def build_reason(c: Dict[str, Any], sc: Dict[str, Any], rank: int, variant: int) -> str:
    p = c.get("profile", {})
    yoe = p.get("years_of_experience", 0)
    yoe_str = f"{float(yoe):.1f}".rstrip("0").rstrip(".") if yoe else "?"
    title = p.get("current_title", "professional")
    opener = _OPENERS[variant % len(_OPENERS)].format(yoe=yoe_str, title=title)

    comp = sc["components"]
    bits = []
    skills = _ai_skills_present(c)
    if skills:
        bits.append(f"core skills {', '.join(skills)}")
    if comp.get("product", 0) > 0.7:
        bits.append("product-company background")
    if comp.get("semantic", 0) > 0.7:
        bits.append("strong semantic match to the JD")
    elif comp.get("semantic", 0) > 0.5:
        bits.append("moderate semantic match")

    sentence = opener + (("; " + "; ".join(bits)) if bits else "") + "."
    concern = _concern(c, sc)
    if concern:
        sentence += f" Concern: {concern}."
    return sentence[:240]
