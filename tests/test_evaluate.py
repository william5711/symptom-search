import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_ddxplus_queries import describe, location_phrase, to_sentence
from evaluate import QUERY_SETS, answer_position, bootstrap_interval, hit_rate, load_queries, reciprocal_ranks, wilson_interval


def test_answer_position_uses_the_best_placed_label():
    assert answer_position(["a", "b", "c"], ["c", "b"]) == 2


def test_answer_position_is_none_without_a_match():
    assert answer_position(["a", "b"], ["z"]) is None
    assert answer_position(["a", "b"], []) is None


def test_hit_rate_and_reciprocal_ranks_count_misses():
    positions = [1, 4, None, 2]
    assert hit_rate(positions, 3) == 0.5
    assert reciprocal_ranks(positions).tolist() == [1.0, 0.25, 0.0, 0.5]


def test_wilson_interval_matches_a_known_value():
    low, high = wilson_interval(14, 24)
    assert low == pytest.approx(0.388, abs=0.002)
    assert high == pytest.approx(0.755, abs=0.002)


def test_wilson_interval_stays_in_range():
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_bootstrap_interval_contains_the_mean_and_repeats():
    values = np.array([0.0, 1.0] * 50)
    low, high = bootstrap_interval(values)
    assert low < 0.5 < high
    assert (low, high) == bootstrap_interval(values)


@pytest.mark.parametrize("question, sentence", [
    ("Do you have a cough?", "I have a cough."),
    ("Have you been coughing up blood?", "I have been coughing up blood."),
    ("Do you feel your abdomen is bloated?", "I feel my abdomen is bloated."),
    ("Do your symptoms of muscle weakness increase with fatigue?", "My symptoms of muscle weakness increase with fatigue."),
])
def test_to_sentence(question, sentence):
    assert to_sentence(question) == sentence


def test_location_phrase_spells_out_the_side():
    assert location_phrase("flank(R)") == "right flank"
    assert location_phrase("palace") == "palate"
    assert location_phrase("upper chest") == "upper chest"


def test_describe_leaves_out_antecedents_and_empty_pain():
    catalog = {
        "E_1": {"is_antecedent": False, "data_type": "B", "question_en": "Do you have a cough?"},
        "E_2": {"is_antecedent": True, "data_type": "B", "question_en": "Do you have COPD?"},
        "E_55": {"is_antecedent": False, "data_type": "M", "question_en": "", "value_meaning": {
            "V_1": {"en": "nowhere"}, "V_2": {"en": "flank(L)"}}},
    }
    assert describe(["E_1", "E_2", "E_55_@_V_1"], catalog) == "I have a cough."
    assert describe(["E_1", "E_55_@_V_2"], catalog) == "I have pain in my left flank. I have a cough."


@pytest.mark.parametrize("query_set", sorted(QUERY_SETS))
def test_query_files_only_label_entries_in_the_dataset(query_set):
    from symptom_search import load_diseases

    names = {d.name for d in load_diseases()}
    queries = load_queries("all", query_set)
    assert {q["split"] for q in queries} == {"dev", "test"}
    assert all(set(q["labels"]) <= names for q in queries)


def test_dev_and_test_share_no_query():
    for query_set in QUERY_SETS:
        dev = {q["query"] for q in load_queries("dev", query_set)}
        test = {q["query"] for q in load_queries("test", query_set)}
        assert not dev & test
