import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_dataset import DATA_DIR, SOURCES_DIR, build, is_condition_topic, parse_summary, to_csv, validate

GOOD_ROWS = [("Flu", "Cough", "wikidata"), ("Flu", "Fever", "medlineplus+wikidata")]


def test_dataset_matches_what_its_sources_build():
    with open(DATA_DIR / "diseases.csv", newline="", encoding="utf-8") as file:
        assert file.read() == to_csv(build())


def test_saved_sources_match_the_manifest():
    manifest = json.loads((SOURCES_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 3
    for name, entry in manifest.items():
        assert hashlib.sha256((SOURCES_DIR / name).read_bytes()).hexdigest() == entry["sha256"]


def test_real_dataset_passes_validation():
    assert validate(build()) == []


def test_validate_accepts_good_rows():
    assert validate(GOOD_ROWS) == []


def test_validate_finds_each_problem():
    assert validate([("Flu", " ", "wikidata")])
    assert validate([("Flu", "Cough", "wikidata"), ("Flu", "cough", "wikidata")])
    assert validate([("Flu", "Cough", "a_blog")])
    assert validate([("Statins", "Cough", "wikidata"), ("Blood Glucose", "Cough", "wikidata")])


def test_validate_stops_a_large_drop_in_entries():
    assert validate(GOOD_ROWS, previous_entries=2) != []
    assert validate(GOOD_ROWS, previous_entries=1) == []


def test_condition_filter():
    assert is_condition_topic("Flu", ["Infections"])
    assert not is_condition_topic("Statins", ["Drug Therapy"])
    assert not is_condition_topic("Diets", ["Food and Nutrition"])
    assert not is_condition_topic("Pregnancy", ["Women", "Pregnancy and Reproduction"])
    assert is_condition_topic("Malnutrition", ["Food and Nutrition"])


def test_parse_summary_falls_back_to_symptom_words():
    assert parse_summary("<p>Hives are red and sometimes itchy bumps on your skin.</p>") == [
        "Hives are red and sometimes itchy bumps on your skin"
    ]


def test_parse_summary_reads_a_symptom_list():
    summary = "<p>Symptoms include:</p><ul><li>Fever</li><li>Sore throat</li><li>Cough</li></ul>"
    assert parse_summary(summary)[:3] == ["Fever", "Sore throat", "Cough"]
