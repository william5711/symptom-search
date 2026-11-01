import numpy as np
import pytest

from symptom_search.data import Disease
from symptom_search.matcher import MODES, SymptomMatcher

VOCABULARY = ["cough", "fever", "rash", "itch", "headache", "thirst"]


class FakeEncoder:
    """Bag-of-words vectors over a small vocabulary, so tests need no model download."""

    def encode(self, texts):
        vectors = np.zeros((len(texts), len(VOCABULARY) + 1))
        for row, text in enumerate(texts):
            for column, word in enumerate(VOCABULARY):
                vectors[row, column] = text.lower().count(word)
            if not vectors[row].any():
                vectors[row, -1] = 1  # unknown text points away from every symptom
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    encode_cached = encode


DISEASES = [
    Disease("Respiratory", ("Cough", "Fever")),
    Disease("Skin", ("Rash", "Itch")),
    # Unrelated symptoms must not score without a similarity match.
    Disease("Rare", ("Headache", "Thirst")),
]


@pytest.mark.parametrize("mode", MODES)
def test_ranks_the_matching_disease_first(mode):
    matcher = SymptomMatcher(DISEASES, FakeEncoder(), mode=mode)
    assert matcher.search("I have a cough and a fever")[0].disease == "Respiratory"
    assert matcher.search("an itch and a rash")[0].disease == "Skin"


def test_evidence_links_clauses_to_symptoms():
    match = SymptomMatcher(DISEASES, FakeEncoder()).search("a cough and something odd")[0]
    assert [(e.clause, e.symptom) for e in match.evidence] == [("a cough", "Cough")]


def test_unlisted_symptoms_give_no_evidence():
    matches = SymptomMatcher(DISEASES, FakeEncoder()).search("a cough")
    assert next(m for m in matches if m.disease == "Rare").evidence == []


def test_unrecognised_text_returns_nothing():
    assert SymptomMatcher(DISEASES, FakeEncoder()).search("the weather is lovely") == []


def test_a_query_with_no_close_symptom_returns_nothing():
    matcher = SymptomMatcher(DISEASES, FakeEncoder(), min_query_similarity=0.8)
    assert matcher.search("cough fever") == []  # similarity 0.71 to each symptom
    assert matcher.search("a cough")[0].disease == "Respiratory"


def test_negated_symptoms_are_ignored():
    assert SymptomMatcher(DISEASES, FakeEncoder()).search("no fever") == []
    top = SymptomMatcher(DISEASES, FakeEncoder()).search("no cough, a rash")[0]
    assert top.disease == "Skin"


def test_top_k_limits_results():
    assert len(SymptomMatcher(DISEASES, FakeEncoder()).search("a cough", top_k=2)) == 2


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        SymptomMatcher(DISEASES, FakeEncoder(), mode="nope")
