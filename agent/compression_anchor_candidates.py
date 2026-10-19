"""Skip lines that cannot match known anchor patterns; unknown patterns scan in full.

The VCC evaluation suggested this optimization independently of its summary engine.
Only exact known pattern sources are filtered, preserving future-pattern behavior.
"""
from __future__ import annotations

# Every match of each listed expression contains the corresponding literal and
# stays within one newline-delimited line. A changed expression gets a full scan.
_REQUIRED_LITERAL = {
    r"\b[\w./-]+/[\w.-]+\.(?:py|ts|tsx|js|rs|md|yaml|yml|json|toml|sh)\b": "/",
    r"\b(?:fix|feat|docs|refactor|chore|salvage|ent)/[A-Za-z0-9._/-]{3,60}": "/",
    r"@[A-Za-z0-9-]{3,30}\b": "@",
    r"https?://[^\s)\"']{10,110}": "://",
}


def anchor_candidate_text(text: str, pattern_source: str) -> str:
    """Preserve all possible matches, their order and their exact line contents."""
    literal = _REQUIRED_LITERAL.get(pattern_source)
    if literal is None:
        return text
    if literal not in text:
        return ""
    lines = text.split("\n")
    selected = [line for line in lines if literal in line]
    # Dense input already needs a full scan; avoid creating another large string.
    return text if len(selected) * 2 >= len(lines) else "\n".join(selected)
