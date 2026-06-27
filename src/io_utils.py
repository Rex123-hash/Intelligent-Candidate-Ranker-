"""Streaming JSONL loader and profile-text construction for embedding."""
import json
from pathlib import Path
from typing import Iterator, Dict, Any

def iter_candidates(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield one candidate dict per non-blank JSONL line, in file order."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def build_profile_text(c: Dict[str, Any]) -> str:
    """Concatenate the semantically meaningful fields into one string for embedding."""
    p = c.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        f"Current role: {p.get('current_title','')} at {p.get('current_company','')} "
        f"({p.get('current_industry','')}).",
    ]
    for role in c.get("career_history", []):
        parts.append(f"{role.get('title','')} at {role.get('company','')}: {role.get('description','')}")
    skills = ", ".join(s.get("name", "") for s in c.get("skills", []))
    if skills:
        parts.append("Skills: " + skills)
    return "\n".join(part for part in parts if part).strip()
