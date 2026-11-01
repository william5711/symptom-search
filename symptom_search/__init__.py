"""Semantic search over a symptom-to-disease knowledge base."""

from .data import Disease, load_diseases
from .matcher import ClauseMatch, Match, SymptomMatcher

__all__ = ["ClauseMatch", "Disease", "Match", "SymptomMatcher", "load_diseases"]
