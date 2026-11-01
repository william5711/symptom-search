"""Break free-form patient text into clauses that each describe one complaint."""

from __future__ import annotations

import re
from typing import List

_CLAUSE_BOUNDARY = re.compile(r"[.!?;,\n]+|\b(?:and|but|with|along with|plus|also)\b", re.IGNORECASE)

# Only explicit denials. A broader rule would wrongly drop real complaints such as
# "lost weight without trying", "no energy" or "can't sleep".
_NEGATION = re.compile(
    r"^(?:no|without)\b(?!\s+(?:energy|appetite|matter|interest|strength|sleep))"
    r"|\b(?:don't|do not|didn't|did not|haven't|have not|hasn't|has not|never)\s+(?:have|had|got)\b"
    r"|\bden(?:y|ies)\b",
    re.IGNORECASE,
)


def split_clauses(text: str) -> List[str]:
    """Split on punctuation and conjunctions, keeping non-empty clauses."""
    return [part.strip() for part in _CLAUSE_BOUNDARY.split(text) if part.strip()]


def is_negated(clause: str) -> bool:
    """True for clauses that deny a symptom, such as "no fever"."""
    return bool(_NEGATION.search(clause.replace("’", "'")))
