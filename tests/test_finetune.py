import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from finetune import batches, contrastive_loss, training_pairs

ROWS = [
    {"disease": "Flu", "symptom": "Fever", "sources": "wikidata"},
    {"disease": "Flu", "symptom": "Cough", "sources": "wikidata"},
    {"disease": "Flu", "symptom": "Feeling hot", "sources": "medlineplus"},
    {"disease": "Held out", "symptom": "Rash", "sources": "wikidata"},
    {"disease": "Held out", "symptom": "Itch", "sources": "wikidata"},
    {"disease": "Lone", "symptom": "Thirst", "sources": "wikidata"},
]


def test_training_pairs_skip_excluded_and_single_symptom_entries():
    pairs = training_pairs(ROWS, {"Held out"}, random.Random(0))
    assert set(pairs) == {"Flu"}


def test_training_pairs_put_cross_source_pairs_first():
    pairs = training_pairs(ROWS, set(), random.Random(0))["Flu"]
    assert all("Feeling hot" in pair for pair in pairs[:2])
    assert "Feeling hot" not in pairs[2]


def test_a_batch_never_has_two_pairs_from_one_entry():
    pairs = {f"entry {i}": [(f"a{i}{j}", f"b{i}{j}") for j in range(i + 1)] for i in range(6)}
    result = batches(pairs, 3, random.Random(0))
    assert result
    for batch in result:
        assert len(batch) == 3
        assert len({a[1] for a, _ in batch}) == 3


def test_contrastive_loss_is_lower_when_pairs_line_up():
    torch = pytest.importorskip("torch")
    left = torch.eye(4)
    assert contrastive_loss(left, left) < contrastive_loss(left, left.flip(0))
