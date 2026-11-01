import pytest

from symptom_search.text import is_negated, split_clauses


def test_split_clauses_on_punctuation_and_conjunctions():
    assert split_clauses("I have a cough, a fever and chills. My head hurts") == [
        "I have a cough",
        "a fever",
        "chills",
        "My head hurts",
    ]


def test_split_clauses_of_blank_text():
    assert split_clauses("  ...  ") == []


@pytest.mark.parametrize("clause", ["no fever", "I don't have a cough", "I don’t have a cough", "denies chest pain"])
def test_denials_are_negated(clause):
    assert is_negated(clause)


@pytest.mark.parametrize(
    "clause",
    ["I've lost weight without trying", "I have no energy", "no appetite", "I can't sleep", "not sleeping well"],
)
def test_complaints_with_negative_words_are_kept(clause):
    assert not is_negated(clause)
